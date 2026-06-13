"""FLARE — monkey-patch pi0_pytorch.sample_actions to support:
  1. SDE injection via score-from-velocity (Tweedie identity for linear paths)
  2. Verifier-guided sampling via DPS-style gradient on Tweedie x̂_0

Sampler control via model attributes (set before policy.infer):
    model.flare_eta          : float, SDE noise level (0 = ODE)
    model.flare_verifier_fn  : callable (obs_features, chunk) -> scalar logit
    model.flare_alpha        : float, guidance scale (0 = no guidance)
    model.flare_obs_state    : torch.Tensor (8,), the proprioceptive obs for the verifier

If flare_eta == 0 AND no guidance, delegates to the original sample_actions
(bit-identical to pre-patch). Otherwise runs the integration loop manually.

Tweedie identity (pi0's convention, v_t = noise - actions):
  x_t       = t·noise + (1-t)·actions
  E[noise|x_t]   = x_t + (1-t)·v_t
  E[actions|x_t] = x_t - t·v_t                 ← Tweedie clean estimate
  score(x_t,t)   = -E[noise|x_t]/t = -(x_t + (1-t)v_t)/t

SDE drift (reverse-time, dt < 0):
  drift = v_t - 0.5·η²·score - α·∇_{x_t} V(x̂_0)
  x_{t+dt} = x_t + dt·drift + η·sqrt(|dt|)·dW

The +α·∇V appears with NEGATIVE sign in drift because dt<0 and we want x to
move toward higher V each step (dt·(-α·∇V) = +α·|dt|·∇V).

DPS approximation: treat v_t as constant w.r.t. x_t so ∂x̂_0/∂x_t ≈ I.
"""
import importlib
import torch


def apply_patch():
    mod = importlib.import_module("openpi.models_pytorch.pi0_pytorch")

    Pi0 = None
    for name in ["PI0Pytorch", "Pi0Pytorch", "PI0", "Pi0Model", "Pi0"]:
        if hasattr(mod, name) and isinstance(getattr(mod, name), type):
            Pi0 = getattr(mod, name)
            break
    if Pi0 is None:
        classes = [n for n in dir(mod) if isinstance(getattr(mod, n), type)]
        raise RuntimeError(f"Could not find pi0 class. Available classes: {classes}")
    print(f"[patch_pi0_sde] patching class: {Pi0.__name__}")

    make_att_2d_masks = getattr(mod, "make_att_2d_masks")
    original_sample = Pi0.sample_actions

    def sample_actions_sde(self, device, observation, noise=None, num_steps=10, eta=None) -> torch.Tensor:
        """SDE + guidance-enabled drop-in for sample_actions.

        eta priority: explicit arg > self.flare_eta > 0.0 (pure ODE).
        Pure ODE without guidance (eta=0.0, alpha=0) delegates to the original.

        Optional hybrid η schedule via:
            self.flare_eta_high       : float, η for early integration (t > threshold)
            self.flare_eta_low        : float, η for late integration (t <= threshold)
            self.flare_eta_threshold  : float, the t value separating high/low (default 0.5)

        If both flare_eta_high and flare_eta_low are set, the schedule overrides
        the scalar eta. Useful for time-localized noise injection where the noise
        spike happens during the commitment window (early t) and execution is
        refined cleanly under low/zero noise (late t).
        """
        if eta is None:
            eta = float(getattr(self, "flare_eta", 0.0))

        # Optional hybrid η schedule
        eta_high = getattr(self, "flare_eta_high", None)
        eta_low = getattr(self, "flare_eta_low", None)
        eta_threshold = float(getattr(self, "flare_eta_threshold", 0.5))
        use_eta_schedule = (eta_high is not None) and (eta_low is not None)
        if use_eta_schedule:
            eta_high = float(eta_high)
            eta_low = float(eta_low)

        # Optional DIRECTED NOISE bias (Track 2) — breaks Tweedie invariance.
        # When flare_noise_bias_direction is set (shape: action_horizon, action_dim),
        # we add bias_strength * direction to the Gaussian noise during the
        # high-noise window. This pushes integration toward a specified mode.
        noise_bias_dir = getattr(self, "flare_noise_bias_direction", None)
        noise_bias_strength = float(getattr(self, "flare_noise_bias_strength", 0.0))
        # Only active during high-noise window — when time > threshold (during a schedule),
        # or always if no schedule (fall back to pure scalar eta).
        use_directed_noise = (noise_bias_dir is not None) and (noise_bias_strength > 0)

        alpha = float(getattr(self, "flare_alpha", 0.0))
        verifier_fn = getattr(self, "flare_verifier_fn", None)
        obs_state = getattr(self, "flare_obs_state", None)
        use_guidance = (alpha != 0.0) and (verifier_fn is not None) and (obs_state is not None)

        # Delegate to original only if there's truly no SDE work and no guidance and no schedule.
        if eta == 0.0 and not use_guidance and not use_eta_schedule:
            # Bit-identical to pre-patch ODE
            return original_sample(self, device, observation, noise=noise, num_steps=num_steps)

        # ---------- SDE / guided path (eta > 0 OR guidance enabled) ----------
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        abs_dt_sqrt = torch.sqrt(torch.abs(dt))
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        # Pre-build obs_features tensor for guidance, if enabled
        if use_guidance:
            # obs_state should be (8,) — broadcast to (B, 8)
            obs_features_g = obs_state.to(device).float()
            if obs_features_g.dim() == 1:
                obs_features_g = obs_features_g.unsqueeze(0)
            if obs_features_g.shape[0] != bsize:
                obs_features_g = obs_features_g.expand(bsize, -1)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            # Velocity (no grad through pi0 — DPS approximation)
            with torch.no_grad():
                v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)

            # Score-from-velocity (Tweedie):
            denom = time.clamp(min=1e-3)
            score = -(x_t + (1.0 - time) * v_t) / denom

            # Guidance gradient on Tweedie x̂_0 = x_t - t·v_t
            grad_V = torch.zeros_like(x_t)
            if use_guidance:
                # DPS approximation: x̂_0 depends linearly on x_t (treat v_t constant)
                x_hat_0 = (x_t - time * v_t).detach().requires_grad_(True)
                with torch.enable_grad():
                    v_score = verifier_fn(obs_features_g, x_hat_0)  # (B,) logits
                    # Sum reduction → scalar; grad gives per-sample grad via x_hat_0
                    grad_V = torch.autograd.grad(v_score.sum(), x_hat_0,
                                                 create_graph=False)[0]
                # Optional: normalize gradient to unit per-sample L2 norm to make
                # alpha scale-invariant w.r.t. verifier output magnitude.
                # (Skip for now — tune alpha directly. Add later if needed.)

            # Determine current η: schedule if active, else scalar.
            # time goes from 1.0 down to 0.0; "high η" applies to early integration
            # where mode commitment happens (the juggle window).
            if use_eta_schedule:
                current_eta = eta_high if time.item() > eta_threshold else eta_low
            else:
                current_eta = eta

            # Directional bias check (window matches the high-η schedule)
            in_high_noise_window = (not use_eta_schedule) or (time.item() > eta_threshold)
            apply_directed_bias = use_directed_noise and in_high_noise_window

            # Compute and pad the bias tensor once if needed
            drift_bias = None
            if apply_directed_bias:
                # noise_bias_dir shape: (action_horizon, action_dim_real) e.g. (50, 7)
                # Pi0's internal action_dim is padded to e.g. 32. Pad to match.
                bias = noise_bias_dir.to(x_t.device, dtype=x_t.dtype)
                if bias.dim() == 2:
                    if bias.shape[-1] < x_t.shape[-1]:
                        pad_size = x_t.shape[-1] - bias.shape[-1]
                        padding = torch.zeros(bias.shape[0], pad_size,
                                              device=bias.device, dtype=bias.dtype)
                        bias = torch.cat([bias, padding], dim=-1)
                    bias = bias.unsqueeze(0).expand_as(x_t)
                drift_bias = noise_bias_strength * bias

            # Drift: deterministic velocity − score correction − guidance + bias
            # The bias is in the DRIFT channel (not noise) so it doesn't get scaled
            # by η · sqrt(|dt|) — it gets the full dt multiplier and is independent
            # of η. Per-step displacement = dt · bias_strength · direction.
            drift = v_t - 0.5 * (current_eta * current_eta) * score - alpha * grad_V
            if drift_bias is not None:
                drift = drift + drift_bias

            # Stochastic step (noise only if eta>0; else pure guided ODE).
            if current_eta > 0.0:
                noise_dW = torch.randn_like(x_t)
                x_t = x_t + dt * drift + current_eta * abs_dt_sqrt * noise_dW
            else:
                x_t = x_t + dt * drift

            time = time + dt

        return x_t

    Pi0.sample_actions = sample_actions_sde
    print(f"[patch_pi0_sde] installed sample_actions_sde + guidance support")
    print(f"[patch_pi0_sde]   - eta=0 + no guidance ≡ ODE (delegates to original)")
    print(f"[patch_pi0_sde]   - eta>0 → SDE")
    print(f"[patch_pi0_sde]   - flare_verifier_fn + flare_alpha → DPS-style guidance")
    return Pi0
