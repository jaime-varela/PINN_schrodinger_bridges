import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def make_density_movie(
    model,
    cfg,
    npts=200,
    t0=0.0,
    t1=1.0,
    num_frames=100,
    fps=20,
    outfile="density.gif",   # default now gif
    writer="pillow",         # "ffmpeg" for .mp4, or "pillow" for .gif
    cmap="plasma",           # customize or leave default
    levels=50,               # contour levels
    with_contours=False,     # True=contourf, False=imshow (faster)
    clamp_min=None           # e.g., 0.0 to clip negative values if needed
):
    device = cfg.device

    # Spatial grid (computed once)
    xline = torch.linspace(-5.0, 5.0, npts, device=device)
    yline = torch.linspace(-5.0, 5.0, npts, device=device)
    xgrid, ygrid = torch.meshgrid(xline, yline, indexing='ij')
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    # Discretize time
    ts = torch.linspace(t0, t1, num_frames, device=device)

    # 1) Pass over time to get global vmin/vmax for fixed color scaling
    with torch.no_grad():
        zmins, zmaxs = [], []
        for t in ts:
            tcol = torch.full((xyinput.shape[0], 1), float(t.item()), device=device)
            phi_t, hphi_t = model(xyinput, tcol)
            prod_t = phi_t * hphi_t
            if clamp_min is not None:
                prod_t = torch.clamp_min(prod_t, clamp_min)
            z = prod_t.squeeze().reshape(npts, npts)
            zmins.append(z.min().item())
            zmaxs.append(z.max().item())

    vmin, vmax = float(np.min(zmins)), float(np.max(zmaxs))
    if np.isclose(vmin, vmax):  # Edge case: prevent zero range
        vmax = vmin + 1e-8

    # 2) Set up the figure
    fig, ax = plt.subplots(figsize=(6, 5))
    X = xgrid.detach().cpu().numpy()
    Y = ygrid.detach().cpu().numpy()

    # First frame
    with torch.no_grad():
        t0col = torch.full((xyinput.shape[0], 1), float(ts[0].item()), device=device)
        phi0, hphi0 = model(xyinput, t0col)
        prod0 = phi0 * hphi0
        if clamp_min is not None:
            prod0 = torch.clamp_min(prod0, clamp_min)
        Z0 = prod0.squeeze().reshape(npts, npts).detach().cpu().numpy()

    if with_contours:
        quad = ax.contourf(X, Y, Z0, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        cbar = fig.colorbar(quad, ax=ax, label='Density')
    else:
        quad = ax.imshow(
            Z0,
            origin='lower',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            aspect='equal',
            interpolation='nearest'
        )
        cbar = fig.colorbar(quad, ax=ax, label='Density')

    ax.set_title(f"Density, t={ts[0].item():.3f}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()

    # 3) Animation update
    def update(frame_idx):
        t = ts[frame_idx]
        with torch.no_grad():
            tcol = torch.full((xyinput.shape[0], 1), float(t.item()), device=device)
            phi_t, hphi_t = model(xyinput, tcol)
            prod_t = phi_t * hphi_t
            if clamp_min is not None:
                prod_t = torch.clamp_min(prod_t, clamp_min)
            Z = prod_t.squeeze().reshape(npts, npts).detach().cpu().numpy()

        if with_contours:
            # Clear and redraw contours each frame
            ax.collections.clear()
            cf = ax.contourf(X, Y, Z, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            quad.set_data(Z)

        ax.set_title(f"Density, t={t.item():.3f}")
        return (quad,)

    anim = FuncAnimation(
        fig, update,
        frames=num_frames,
        interval=1000.0 / fps,
        blit=not with_contours
    )

    # 4) Write to disk
    writer_lower = writer.lower()
    saved_path = outfile

    if writer_lower == "ffmpeg":
        # Ensure mp4 extension
        if not saved_path.lower().endswith(".mp4"):
            saved_path = saved_path.rsplit(".", 1)[0] + ".mp4"
        try:
            anim.save(saved_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
        except Exception as e:
            print(f"FFmpeg failed with: {e}\nFalling back to GIF via PillowWriter.")
            saved_path = saved_path.rsplit(".", 1)[0] + ".gif"
            anim.save(saved_path, writer=PillowWriter(fps=fps))
            print(f"Saved {saved_path}")
    elif writer_lower == "pillow":
        # Ensure gif extension
        if not saved_path.lower().endswith(".gif"):
            saved_path = saved_path.rsplit(".", 1)[0] + ".gif"
        anim.save(saved_path, writer=PillowWriter(fps=fps))
    else:
        raise ValueError("writer must be 'ffmpeg' or 'pillow'")

    plt.close(fig)
    return saved_path
