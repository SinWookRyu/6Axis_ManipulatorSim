"""Main GUI window for the 6-DOF manipulator simulator."""

import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from kinematics import (
    DHParameters, forward_kinematics, get_joint_positions,
    rotation_matrix_to_rpy, rpy_to_rotation_matrix,
    geometric_jacobian, ik_fast, inverse_kinematics_dls,
    SingularityDetector,
)
from dynamics import joint_load_report
from workspace import sample_workspace, workspace_stats


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG        = "#1e1e2e"
FG        = "#cdd6f4"
ACCENT    = "#89b4fa"
WARN      = "#fab387"
ERR       = "#f38ba8"
OK_COL    = "#a6e3a1"
PANEL_BG  = "#181825"
ENTRY_BG  = "#313244"

LINK_COLORS = ["#4fc3f7", "#4fc3f7", "#80cbc4", "#ffb74d", "#ffb74d", "#ef9a9a"]
JOINT_COLOR = "#ffd54f"
EE_COLORS   = ["#ef5350", "#66bb6a", "#42a5f5"]   # X=red, Y=green, Z=blue


class ManipulatorApp:
    """Top-level application class."""

    # Base speeds (per update tick at ~30 fps)
    _BASE_TRANS_MM = 2.0   # mm per tick at speed=1.0
    _BASE_ROT_DEG  = 1.0   # deg per tick at speed=1.0
    _UPDATE_MS     = 33    # ~30 fps

    # Key → action mapping
    _KEY_MAP = {
        "a": "x-", "d": "x+",
        "w": "y+", "s": "y-",
        "r": "z+", "f": "z-",
        "q": "roll-", "e": "roll+",
        "i": "pitch+", "k": "pitch-",
        "j": "yaw-",  "l": "yaw+",
    }

    def __init__(self, root: tk.Tk, dh_params: DHParameters):
        self.root = root
        self.dh_params = dh_params

        self.joint_angles = np.zeros(len(dh_params))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.external_wrench = np.zeros(6)
        self.active_keys: set = set()

        self._sing_detector = SingularityDetector()
        self._workspace_pts: np.ndarray | None = None
        self._show_workspace = False
        self._workspace_busy = False

        self._build_ui()
        self._bind_keys()
        self._redraw()
        self._update_state_displays()
        self._schedule_update()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("6-DOF Manipulator Simulator")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 720)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=PANEL_BG, foreground=FG,
                        fieldbackground=ENTRY_BG, bordercolor="#45475a",
                        troughcolor="#313244", selectbackground=ACCENT)
        style.configure("TNotebook", background=PANEL_BG)
        style.configure("TNotebook.Tab", background="#313244", foreground=FG,
                        padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])
        style.configure("TLabel",  background=PANEL_BG, foreground=FG)
        style.configure("TFrame",  background=PANEL_BG)
        style.configure("TButton", background="#45475a", foreground=FG,
                        padding=[6, 3])
        style.configure("TLabelframe", background=PANEL_BG, foreground=ACCENT)
        style.configure("TLabelframe.Label", background=PANEL_BG, foreground=ACCENT)
        style.configure("TScale", background=PANEL_BG, troughcolor="#313244")

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._sim_tab = ttk.Frame(nb)
        self._dh_tab  = ttk.Frame(nb)
        nb.add(self._sim_tab, text=" Simulation ")
        nb.add(self._dh_tab,  text=" DH Parameters ")

        self._build_sim_tab()
        self._build_dh_tab()

        # Status bar
        self._status_var = tk.StringVar(value="Ready")
        self._status_color = tk.StringVar(value=OK_COL)
        sf = tk.Frame(self.root, bg=PANEL_BG, height=24)
        sf.pack(fill=tk.X, side=tk.BOTTOM)
        self._status_lbl = tk.Label(
            sf, textvariable=self._status_var, anchor=tk.W,
            bg=PANEL_BG, fg=OK_COL, font=("Consolas", 9), padx=6,
        )
        self._status_lbl.pack(fill=tk.X)

    # ── Simulation tab ──────────────────────────────────────────────────────

    def _build_sim_tab(self):
        pane = ttk.PanedWindow(self._sim_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        view_frame = ttk.Frame(pane)
        ctrl_frame = ttk.Frame(pane, width=290)
        pane.add(view_frame,  weight=4)
        pane.add(ctrl_frame,  weight=1)

        self._build_3d_view(view_frame)
        self._build_control_panel(ctrl_frame)

    def _build_3d_view(self, parent):
        self._fig = Figure(figsize=(8, 7), facecolor=BG)
        self._ax  = self._fig.add_subplot(111, projection="3d", facecolor=BG)
        self._style_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        widget = self._canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        widget.configure(bg=BG)

        # Give focus back to root on canvas click so key bindings work
        widget.bind("<Button-1>", lambda e: self.root.focus_set())

        toolbar_frame = tk.Frame(parent, bg=BG)
        toolbar_frame.pack(fill=tk.X)
        tb = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        tb.config(background=BG)
        tb.update()

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(BG)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#45475a")
        ax.tick_params(colors="#585b70", labelsize=7)
        ax.xaxis.label.set_color("#585b70")
        ax.yaxis.label.set_color("#585b70")
        ax.zaxis.label.set_color("#585b70")
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)

    def _build_control_panel(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", on_frame_configure)

        def on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)
        canvas.bind("<Configure>", on_canvas_configure)

        p = 4
        # ── Keyboard shortcuts help ─
        kf = ttk.LabelFrame(inner, text="Keyboard Control (EE Space)", padding=p)
        kf.pack(fill=tk.X, pady=3, padx=4)
        help_text = (
            "A/D  → X-/+    W/S  → Y+/-\n"
            "R/F  → Z+/-    Q/E  → Roll-/+\n"
            "I/K  → Pitch+/−  J/L  → Yaw-/+"
        )
        tk.Label(kf, text=help_text, bg=PANEL_BG, fg="#a6adc8",
                 font=("Consolas", 8), justify=tk.LEFT).pack(anchor=tk.W)

        # ── Speed ─
        sf = ttk.LabelFrame(inner, text="Speed", padding=p)
        sf.pack(fill=tk.X, pady=3, padx=4)
        tk.Scale(sf, variable=self.speed_var, from_=0.1, to=5.0,
                 resolution=0.1, orient=tk.HORIZONTAL, bg=PANEL_BG, fg=FG,
                 highlightthickness=0, troughcolor="#313244", activebackground=ACCENT,
                 label="").pack(fill=tk.X)
        ttk.Label(sf, textvariable=self.speed_var).pack(anchor=tk.E)

        # ── Joint space ─
        jf = ttk.LabelFrame(inner, text="Joint Space (°)", padding=p)
        jf.pack(fill=tk.X, pady=3, padx=4)
        self._joint_vars = []
        self._joint_sliders = []
        for i in range(len(self.dh_params.joints)):
            row = ttk.Frame(jf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"J{i+1}:", width=3).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self._joint_vars.append(var)
            joint = self.dh_params.joints[i]
            sl = tk.Scale(
                row, variable=var, orient=tk.HORIZONTAL,
                from_=np.degrees(joint.theta_min),
                to=np.degrees(joint.theta_max),
                resolution=0.1, bg=PANEL_BG, fg=FG,
                highlightthickness=0, troughcolor="#313244",
                activebackground=ACCENT, showvalue=False,
                command=lambda val, idx=i: self._on_joint_slider(idx, float(val)),
            )
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._joint_sliders.append(sl)
            val_lbl = ttk.Label(row, textvariable=var, width=8)
            val_lbl.pack(side=tk.RIGHT)

        # ── World space ─
        wf = ttk.LabelFrame(inner, text="World Space (EE Pose)", padding=p)
        wf.pack(fill=tk.X, pady=3, padx=4)
        self._world_vars = {}
        for label in ("X(mm)", "Y(mm)", "Z(mm)", "Roll(°)", "Pitch(°)", "Yaw(°)"):
            row = ttk.Frame(wf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{label}:", width=8).pack(side=tk.LEFT)
            v = tk.StringVar(value="0.000")
            self._world_vars[label] = v
            ttk.Label(row, textvariable=v, foreground=ACCENT,
                      font=("Consolas", 9)).pack(side=tk.LEFT)

        # ── External force/torque ─
        ef = ttk.LabelFrame(inner, text="External Wrench at EE", padding=p)
        ef.pack(fill=tk.X, pady=3, padx=4)
        self._wrench_entries = []
        labels = ["Fx(N)", "Fy(N)", "Fz(N)", "Mx(Nm)", "My(Nm)", "Mz(Nm)"]
        for lbl in labels:
            row = ttk.Frame(ef)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"{lbl}:", width=7).pack(side=tk.LEFT)
            e = ttk.Entry(row, width=10)
            e.insert(0, "0.0")
            e.pack(side=tk.LEFT)
            self._wrench_entries.append(e)
        ttk.Button(ef, text="Apply Force", command=self._apply_external_force).pack(pady=3)

        # ── Joint torques ─
        tf = ttk.LabelFrame(inner, text="Joint Torques (N·m / %rated)", padding=p)
        tf.pack(fill=tk.X, pady=3, padx=4)
        self._torque_vars = []
        self._torque_bars = []
        for i in range(len(self.dh_params.joints)):
            row = ttk.Frame(tf)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"J{i+1}:", width=3).pack(side=tk.LEFT)
            v = tk.StringVar(value="  0.00 Nm (  0%)")
            self._torque_vars.append(v)
            ttk.Label(row, textvariable=v, font=("Consolas", 8),
                      foreground="#cba6f7").pack(side=tk.LEFT)

        # ── Workspace ─
        wsf = ttk.LabelFrame(inner, text="Workspace Visualisation", padding=p)
        wsf.pack(fill=tk.X, pady=3, padx=4)
        self._ws_status_var = tk.StringVar(value="Not computed")
        ttk.Label(wsf, textvariable=self._ws_status_var,
                  font=("Consolas", 8), foreground="#a6adc8").pack(anchor=tk.W)
        btn_row = ttk.Frame(wsf)
        btn_row.pack(fill=tk.X, pady=2)
        self._ws_btn = ttk.Button(btn_row, text="Compute Workspace",
                                  command=self._compute_workspace_async)
        self._ws_btn.pack(side=tk.LEFT, padx=2)
        self._ws_toggle_btn = ttk.Button(btn_row, text="Show",
                                         command=self._toggle_workspace,
                                         state=tk.DISABLED)
        self._ws_toggle_btn.pack(side=tk.LEFT, padx=2)

        # ── Buttons ─
        bf = ttk.Frame(inner)
        bf.pack(fill=tk.X, pady=6, padx=4)
        ttk.Button(bf, text="Reset Home", command=self._reset_home).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(bf, text="Zero Joints", command=self._zero_joints).pack(
            side=tk.LEFT, padx=2)

    # ── DH parameters tab ───────────────────────────────────────────────────

    def _build_dh_tab(self):
        outer = ttk.Frame(self._dh_tab, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer,
                  text="Edit DH parameters and press Apply. Angles in degrees, lengths in mm.",
                  foreground="#a6adc8", font=("Consolas", 9)).pack(anchor=tk.W, pady=(0, 6))

        headers = ["Joint", "a (mm)", "d (mm)", "α (°)", "θ_offset (°)", "θ_min (°)", "θ_max (°)"]
        hf = ttk.Frame(outer)
        hf.pack(fill=tk.X)
        for j, h in enumerate(headers):
            ttk.Label(hf, text=h, foreground=ACCENT, font=("Consolas", 9, "bold"),
                      width=12, anchor=tk.CENTER).grid(row=0, column=j, padx=3, pady=2)

        self._dh_entries: list[list[tk.StringVar]] = []
        for i, joint in enumerate(self.dh_params.joints):
            row_vars = []
            values = [
                joint.name or f"J{i+1}",
                f"{joint.a:.2f}",
                f"{joint.d:.2f}",
                f"{np.degrees(joint.alpha):.2f}",
                f"{np.degrees(joint.theta_offset):.2f}",
                f"{np.degrees(joint.theta_min):.2f}",
                f"{np.degrees(joint.theta_max):.2f}",
            ]
            for j, val in enumerate(values):
                v = tk.StringVar(value=str(val))
                row_vars.append(v)
                state = "disabled" if j == 0 else "normal"
                ent = ttk.Entry(hf, textvariable=v, width=12, state=state)
                ent.grid(row=i + 1, column=j, padx=3, pady=2)
            self._dh_entries.append(row_vars)

        btn_row = ttk.Frame(outer)
        btn_row.pack(anchor=tk.W, pady=8)
        ttk.Button(btn_row, text="Apply DH Parameters",
                   command=self._apply_dh_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Reset to FANUC M-10iA",
                   command=self._reset_fanuc_dh).pack(side=tk.LEFT, padx=4)

    # -----------------------------------------------------------------------
    # Keyboard bindings
    # -----------------------------------------------------------------------

    def _bind_keys(self):
        self.root.bind("<KeyPress>",   self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key in self._KEY_MAP:
            self.active_keys.add(self._KEY_MAP[key])

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in self._KEY_MAP:
            self.active_keys.discard(self._KEY_MAP[key])

    # -----------------------------------------------------------------------
    # Update loop
    # -----------------------------------------------------------------------

    def _schedule_update(self):
        self.root.after(self._UPDATE_MS, self._update_loop)

    def _update_loop(self):
        try:
            if self.active_keys:
                self._process_movement()
        finally:
            self._schedule_update()

    def _process_movement(self):
        speed = self.speed_var.get()
        step_t = self._BASE_TRANS_MM * speed
        step_r = np.radians(self._BASE_ROT_DEG * speed)

        dp = np.zeros(3)
        drpy = np.zeros(3)

        delta_map = {
            "x+": (dp, 0, +step_t), "x-": (dp, 0, -step_t),
            "y+": (dp, 1, +step_t), "y-": (dp, 1, -step_t),
            "z+": (dp, 2, +step_t), "z-": (dp, 2, -step_t),
            "roll+":  (drpy, 0, +step_r), "roll-":  (drpy, 0, -step_r),
            "pitch+": (drpy, 1, +step_r), "pitch-": (drpy, 1, -step_r),
            "yaw+":   (drpy, 2, +step_r), "yaw-":   (drpy, 2, -step_r),
        }
        for action in list(self.active_keys):
            if action in delta_map:
                arr, idx, val = delta_map[action]
                arr[idx] += val

        _, T_ee = forward_kinematics(self.dh_params, self.joint_angles)
        T_target = T_ee.copy()
        T_target[:3, 3] += dp
        if np.any(drpy != 0):
            R_delta = rpy_to_rotation_matrix(*drpy)
            T_target[:3, :3] = R_delta @ T_ee[:3, :3]

        q_new, success, err = ik_fast(
            self.dh_params, T_target, q_init=self.joint_angles
        )

        if success or err < 5.0:
            self.joint_angles = q_new
            self._sync_sliders()
            self._redraw()
            self._update_state_displays()

    def _on_joint_slider(self, idx: int, deg_val: float):
        """Called when a joint slider is moved directly."""
        self.joint_angles[idx] = np.radians(deg_val)
        self._redraw()
        self._update_state_displays()

    # -----------------------------------------------------------------------
    # 3D rendering
    # -----------------------------------------------------------------------

    def _redraw(self):
        ax = self._ax
        ax.cla()
        self._style_axes()

        T_list, T_ee = forward_kinematics(self.dh_params, self.joint_angles)
        pts = np.array([T[:3, 3] for T in T_list])

        # Draw workspace points
        if self._show_workspace and self._workspace_pts is not None:
            wp = self._workspace_pts
            step = max(1, len(wp) // 3000)
            ax.scatter(
                wp[::step, 0], wp[::step, 1], wp[::step, 2],
                c="#89b4fa", alpha=0.03, s=1, depthshade=False,
            )

        # Draw base platform
        cx, cy = pts[0, 0], pts[0, 1]
        r = 120
        theta = np.linspace(0, 2 * np.pi, 40)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                np.zeros(40), color="#585b70", linewidth=1.5)
        ax.plot([cx, cx], [cy, cy], [0, pts[0, 2]],
                color="#585b70", linewidth=3, linestyle="--")

        # Draw links
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            col = LINK_COLORS[min(i, len(LINK_COLORS) - 1)]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=col, linewidth=5, solid_capstyle="round", zorder=3)

        # Draw joints
        ax.scatter(
            pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2],
            color=JOINT_COLOR, s=80, zorder=5, depthshade=False,
        )
        # EE
        ax.scatter(*pts[-1], color="#f5c2e7", s=120, marker="*",
                   zorder=6, depthshade=False)

        # EE coordinate frame
        scale = 100.0
        R_ee  = T_ee[:3, :3]
        p_ee  = T_ee[:3, 3]
        for i, col in enumerate(EE_COLORS):
            ax.quiver(
                *p_ee, *(R_ee[:, i] * scale),
                color=col, linewidth=2.0, arrow_length_ratio=0.25,
            )

        # Axes limits based on max reach
        reach = sum(abs(j.a) + abs(j.d) for j in self.dh_params.joints)
        lim = reach * 0.75
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, reach * 1.1)

        self._canvas.draw_idle()

    # -----------------------------------------------------------------------
    # State display updates
    # -----------------------------------------------------------------------

    def _update_state_displays(self):
        self._update_world_display()
        self._update_torque_display()
        self._update_singularity_display()

    def _update_world_display(self):
        _, T_ee = forward_kinematics(self.dh_params, self.joint_angles)
        p = T_ee[:3, 3]
        rpy = np.degrees(rotation_matrix_to_rpy(T_ee[:3, :3]))
        keys = ("X(mm)", "Y(mm)", "Z(mm)", "Roll(°)", "Pitch(°)", "Yaw(°)")
        vals = list(p) + list(rpy)
        for k, v in zip(keys, vals):
            self._world_vars[k].set(f"{v:+.3f}")

    def _update_torque_display(self):
        _, report = joint_load_report(
            self.dh_params, self.joint_angles, self.external_wrench
        )
        for i, entry in enumerate(report):
            tau  = entry["torque_nm"]
            pct  = entry["load_pct"]
            self._torque_vars[i].set(f"{tau:+7.2f} Nm ({pct:5.1f}%)")

    def _update_singularity_display(self):
        result = self._sing_detector.check(self.dh_params, self.joint_angles)
        self._status_var.set(result["label"])
        color_map = {"green": OK_COL, "orange": WARN, "red": ERR}
        col = color_map.get(result["color"], FG)
        self._status_lbl.configure(fg=col)

    def _sync_sliders(self):
        """Sync slider positions to current joint_angles (without triggering callbacks)."""
        for i, v in enumerate(self._joint_vars):
            v.set(round(np.degrees(self.joint_angles[i]), 2))

    # -----------------------------------------------------------------------
    # DH parameter editing
    # -----------------------------------------------------------------------

    def _apply_dh_params(self):
        try:
            new_joints = []
            for i, row_vars in enumerate(self._dh_entries):
                a           = float(row_vars[1].get())
                d           = float(row_vars[2].get())
                alpha       = np.radians(float(row_vars[3].get()))
                theta_off   = np.radians(float(row_vars[4].get()))
                theta_min   = np.radians(float(row_vars[5].get()))
                theta_max   = np.radians(float(row_vars[6].get()))
                name        = row_vars[0].get()
                if theta_min >= theta_max:
                    raise ValueError(f"J{i+1}: θ_min must be < θ_max")
                from kinematics import JointDH
                new_joints.append(JointDH(
                    a=a, d=d, alpha=alpha, theta_offset=theta_off,
                    theta_min=theta_min, theta_max=theta_max, name=name,
                ))
            self.dh_params = DHParameters(joints=new_joints, name=self.dh_params.name)
            self.joint_angles = np.zeros(len(self.dh_params))
            self._workspace_pts = None
            self._show_workspace = False
            self._ws_toggle_btn.configure(state=tk.DISABLED, text="Show")
            self._ws_status_var.set("Not computed (DH changed)")
            self._rebuild_joint_sliders()
            self._redraw()
            self._update_state_displays()
        except Exception as exc:
            messagebox.showerror("DH Parameter Error", str(exc))

    def _rebuild_joint_sliders(self):
        """Recreate joint slider limits after DH change."""
        for i, (sl, joint) in enumerate(
            zip(self._joint_sliders, self.dh_params.joints)
        ):
            sl.configure(
                from_=np.degrees(joint.theta_min),
                to=np.degrees(joint.theta_max),
            )
            self._joint_vars[i].set(0.0)

    def _reset_fanuc_dh(self):
        self.dh_params = DHParameters.fanuc_m10ia()
        self.joint_angles = np.zeros(len(self.dh_params))
        self._workspace_pts = None
        self._show_workspace = False
        self._ws_toggle_btn.configure(state=tk.DISABLED, text="Show")
        self._ws_status_var.set("Not computed")
        for i, row_vars in enumerate(self._dh_entries):
            joint = self.dh_params.joints[i]
            row_vars[1].set(f"{joint.a:.2f}")
            row_vars[2].set(f"{joint.d:.2f}")
            row_vars[3].set(f"{np.degrees(joint.alpha):.2f}")
            row_vars[4].set(f"{np.degrees(joint.theta_offset):.2f}")
            row_vars[5].set(f"{np.degrees(joint.theta_min):.2f}")
            row_vars[6].set(f"{np.degrees(joint.theta_max):.2f}")
        self._rebuild_joint_sliders()
        self._redraw()
        self._update_state_displays()

    # -----------------------------------------------------------------------
    # External force
    # -----------------------------------------------------------------------

    def _apply_external_force(self):
        try:
            vals = [float(e.get()) for e in self._wrench_entries]
            self.external_wrench = np.array(vals)
            self._update_torque_display()
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers for wrench.")

    # -----------------------------------------------------------------------
    # Workspace
    # -----------------------------------------------------------------------

    def _compute_workspace_async(self):
        if self._workspace_busy:
            return
        self._workspace_busy = True
        self._ws_btn.configure(state=tk.DISABLED)
        self._ws_status_var.set("Computing... (please wait)")

        def worker():
            pts = sample_workspace(self.dh_params, n_samples=25_000)
            stats = workspace_stats(pts)
            self._workspace_pts = pts
            self.root.after(0, lambda: self._on_workspace_done(stats))

        threading.Thread(target=worker, daemon=True).start()

    def _on_workspace_done(self, stats: dict):
        self._workspace_busy = False
        self._ws_btn.configure(state=tk.NORMAL)
        self._ws_toggle_btn.configure(state=tk.NORMAL)
        info = (
            f"N={stats['n_samples']:,}  "
            f"Reach: {stats['min_reach']:.0f}–{stats['max_reach']:.0f} mm"
        )
        self._ws_status_var.set(info)

    def _toggle_workspace(self):
        self._show_workspace = not self._show_workspace
        self._ws_toggle_btn.configure(
            text="Hide" if self._show_workspace else "Show"
        )
        self._redraw()

    # -----------------------------------------------------------------------
    # Reset helpers
    # -----------------------------------------------------------------------

    def _reset_home(self):
        """Move to a typical ready position."""
        self.joint_angles = np.radians([0, 0, 0, 0, 0, 0])
        self._sync_sliders()
        self._redraw()
        self._update_state_displays()

    def _zero_joints(self):
        self.joint_angles = np.zeros(len(self.dh_params))
        self._sync_sliders()
        self._redraw()
        self._update_state_displays()
