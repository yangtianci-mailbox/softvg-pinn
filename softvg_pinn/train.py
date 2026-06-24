"""Training loop for SoftVG-PINN."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import copy
import json

import pandas as pd
import torch

from .collocation import generate_pde_colpoints
from .config import TrainingConfig, resolve_device, save_config, set_seed
from .data import dataframe_to_tensors, make_train_eval_split, prepare_observations, read_observation_table
from .metrics import regression_metrics, vg_curve_metrics
from .model import SoilPINN
from .plots import plot_vg_curves, plot_water_content


def _weighted_loss(loss_data, loss_pde, loss_vg, loss_prior, data_w, vg_w, pde_w, prior_w):
    return data_w * loss_data + vg_w * loss_vg + pde_w * loss_pde + prior_w * loss_prior


def _append_log(logs, stage, epoch, loss, loss_data, loss_pde, loss_vg, loss_prior, interface_k, pde_weight):
    row = {
        "stage": stage,
        "epoch": int(epoch),
        "total": float(loss.item()),
        "data": float(loss_data.item()),
        "pde": float(loss_pde.item()),
        "vg": float(loss_vg.item()),
        "prior": float(loss_prior.item()),
        "interface_k": float(interface_k),
        "pde_weight": float(pde_weight),
    }
    logs.append(row)
    print(row)


@torch.no_grad()
def _predict_dataframe(model: SoilPINN, df: pd.DataFrame, config: TrainingConfig, device: torch.device) -> pd.DataFrame:
    x = torch.tensor(df[config.col_x].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    t = torch.tensor(df[config.col_t].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    h, k, theta = model(x, t)
    out = df.copy()
    out["theta_reconstructed"] = theta.detach().cpu().numpy().flatten()
    out["h_reconstructed_cm"] = h.detach().cpu().numpy().flatten()
    out["K_reconstructed_cm_day^-1"] = k.detach().cpu().numpy().flatten()
    return out


def train_model(config: TrainingConfig) -> Dict[str, object]:
    """Train a SoftVG-PINN model and write reproducibility outputs."""
    set_seed(config.seed)
    device = resolve_device(config.device)

    output_dir = Path(config.output_dir) / config.experiment_name
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_observation_table(config.data_path, sheet_name=config.sheet_name)
    df = prepare_observations(raw_df, config)
    train_df, eval_df, metric_label = make_train_eval_split(df, config)
    if train_df.empty:
        raise ValueError("Training set is empty after applying validation split.")
    if eval_df.empty:
        raise ValueError("Evaluation set is empty after applying validation split.")

    x_train, t_train, theta_train = dataframe_to_tensors(train_df, config.col_x, config.col_t, config.col_theta, device)
    x_all, t_all, _ = dataframe_to_tensors(df, config.col_x, config.col_t, config.col_theta, device)

    x_min, x_max = x_train.min().item(), x_train.max().item()
    t_min, t_max = t_train.min().item(), t_train.max().item()

    model = SoilPINN(
        config.layer_tuples,
        net_layers=config.net_layers,
        interface_k=config.interface_k_initial,
        interface_buffer_cm=config.interface_buffer_cm,
    ).float().to(device)

    x_f, t_f = generate_pde_colpoints(
        config.n_collocation, x_min, x_max, t_min, t_max, model.interfaces, device
    )

    logs = []
    weights = config.loss_weights

    if config.stage1_epochs > 0:
        opt1 = torch.optim.Adam(model.parameters(), lr=config.stage1_lr)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=max(config.stage1_epochs, 1), eta_min=1.0e-4
        )
        for epoch in range(config.stage1_epochs):
            if (epoch + 1) % 1000 == 0:
                model.interface_k = min(config.interface_k_stage1_max, model.interface_k + 1.0)
            if (epoch + 1) % config.refresh_collocation_every == 0:
                x_f, t_f = generate_pde_colpoints(
                    config.n_collocation, x_min, x_max, t_min, t_max, model.interfaces, device
                )
            x_pde = torch.cat([x_train, x_f], dim=0)
            t_pde = torch.cat([t_train, t_f], dim=0)

            opt1.zero_grad()
            loss_data, loss_pde, loss_vg = model.compute_losses(x_train, t_train, theta_train, x_pde, t_pde)
            loss_prior = model.compute_prior_loss()
            loss = _weighted_loss(loss_data, loss_pde, loss_vg, loss_prior, weights.data, weights.vg, 0.0, weights.prior)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            opt1.step()
            scheduler1.step()

            if (epoch + 1) % config.print_every == 0 or epoch == config.stage1_epochs - 1:
                _append_log(logs, 1, epoch + 1, loss, loss_data, loss_pde, loss_vg, loss_prior, model.interface_k, 0.0)

    if config.stage2_epochs > 0:
        opt2 = torch.optim.Adam(model.parameters(), lr=config.stage2_lr)
        scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5000, gamma=0.5)
        for epoch in range(config.stage2_epochs):
            if (epoch + 1) % 1000 == 0:
                model.interface_k = min(config.interface_k_stage2_max, model.interface_k + 0.5)
            if (epoch + 1) % config.refresh_collocation_every == 0:
                x_f, t_f = generate_pde_colpoints(
                    config.n_collocation, x_min, x_max, t_min, t_max, model.interfaces, device
                )
            x_pde = torch.cat([x_train, x_f], dim=0)
            t_pde = torch.cat([t_train, t_f], dim=0)

            opt2.zero_grad()
            loss_data, loss_pde, loss_vg = model.compute_losses(x_train, t_train, theta_train, x_pde, t_pde)
            loss_prior = model.compute_prior_loss()
            progress = epoch / max(config.stage2_epochs - 1, 1)
            pde_w = config.pde_weight_start + (config.pde_weight_end - config.pde_weight_start) * progress
            loss = _weighted_loss(loss_data, loss_pde, loss_vg, loss_prior, weights.data, weights.vg, pde_w, weights.prior)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            opt2.step()
            scheduler2.step()

            if (epoch + 1) % config.print_every == 0 or epoch == config.stage2_epochs - 1:
                _append_log(logs, 2, epoch + 1, loss, loss_data, loss_pde, loss_vg, loss_prior, model.interface_k, pde_w)

    if config.use_lbfgs and config.lbfgs_max_iter > 0:
        print("Stage 3: L-BFGS final refinement of the composite objective.")
        opt3 = torch.optim.LBFGS(
            model.parameters(),
            max_iter=config.lbfgs_max_iter,
            history_size=50,
            tolerance_grad=1.0e-7,
            tolerance_change=1.0e-9,
            line_search_fn="strong_wolfe",
        )
        x_pde = torch.cat([x_train, x_f], dim=0)
        t_pde = torch.cat([t_train, t_f], dim=0)
        state = {"iter": 0, "best_loss": float("inf"), "best_state": None}

        def closure():
            opt3.zero_grad()
            loss_data, loss_pde, loss_vg = model.compute_losses(x_train, t_train, theta_train, x_pde, t_pde)
            loss_prior = model.compute_prior_loss()
            loss = _weighted_loss(
                loss_data, loss_pde, loss_vg, loss_prior,
                weights.data, weights.vg, config.pde_weight_end, weights.prior
            )
            loss.backward()
            if loss.item() < state["best_loss"] and not torch.isnan(loss):
                state["best_loss"] = float(loss.item())
                state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state["iter"] += 1
            if state["iter"] % max(1, config.print_every) == 0:
                _append_log(
                    logs, 3, state["iter"], loss, loss_data, loss_pde, loss_vg,
                    loss_prior, model.interface_k, config.pde_weight_end
                )
            return loss

        opt3.step(closure)
        if state["best_state"] is not None:
            model.load_state_dict(state["best_state"])

    predictions = _predict_dataframe(model, df, config, device)
    eval_predictions = predictions.loc[eval_df.index]
    train_predictions = predictions.loc[train_df.index]

    train_metrics = regression_metrics(train_predictions[config.col_theta], train_predictions["theta_reconstructed"])
    eval_metrics = regression_metrics(eval_predictions[config.col_theta], eval_predictions["theta_reconstructed"])

    metrics = {
        "metric_context": metric_label,
        "note": (
            "If metric_context is 'reconstruction', all observations were used for training. "
            "theta metrics are fitting/reconstruction metrics, not independent prediction metrics."
        ),
        "train_theta": train_metrics,
        "eval_theta": eval_metrics,
        "vg_curve_metrics_if_reference_available": vg_curve_metrics(model, config.layer_tuples),
    }

    predictions.to_csv(tables_dir / "predictions.csv", index=False)
    pd.DataFrame(model.get_parameter_table()).to_csv(tables_dir / "learned_vg_parameters.csv", index=False)
    pd.DataFrame(logs).to_csv(tables_dir / "training_log.csv", index=False)
    with (tables_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    save_config(config, output_dir / "config_used.yaml")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(config.metadata, f, indent=2, ensure_ascii=False)

    plot_vg_curves(model, config.layer_tuples, figures_dir)
    available_depths = sorted(predictions[config.col_x].unique())
    plot_depths = available_depths[: min(8, len(available_depths))]
    plot_water_content(predictions, plot_depths, config.col_x, config.col_t, config.col_theta, "theta_reconstructed", figures_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": str(output_dir / "config_used.yaml"),
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )

    print(f"Finished. Results written to: {output_dir}")
    return {"model": model, "metrics": metrics, "output_dir": str(output_dir)}
