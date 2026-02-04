"""
Manim animation: 3D imaging reduces reliance on precise probe orientation.

Render with:
    manim -qm substack/probe_orientation.py ProbeOrientation
"""

from manim import *
import numpy as np


class ProbeOrientation(Scene):
    def construct(self):
        self.scene1_2d_problem()
        self.clear_scene()
        self.scene2_3d_solution()
        self.clear_scene()
        self.scene3_reslicing()
        self.clear_scene()
        self.scene4_tradeoff()

    def clear_scene(self):
        self.play(FadeOut(*self.mobjects), run_time=0.5)
        self.wait(0.3)

    # ------------------------------------------------------------------ #
    # Scene 1: 2D Ultrasound Problem
    # ------------------------------------------------------------------ #
    def scene1_2d_problem(self):
        title = Text("2D Ultrasound", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))

        # Body surface
        skin = Line(LEFT * 5, RIGHT * 5, color=LIGHT_BROWN).shift(UP * 1.5)
        skin_label = Text("skin", font_size=20, color=LIGHT_BROWN).next_to(
            skin, RIGHT, buff=0.2
        )
        self.play(Create(skin), FadeIn(skin_label))

        # Heart shape inside the body
        heart = self._make_heart().scale(0.8).shift(DOWN * 1.2)
        heart_label = Text("heart", font_size=20, color=RED_C).next_to(
            heart, DOWN, buff=0.2
        )
        self.play(FadeIn(heart), FadeIn(heart_label))

        # 2D probe (rectangle on skin)
        probe = Rectangle(width=0.6, height=0.3, color=BLUE, fill_opacity=0.8)
        probe.move_to(skin.get_center() + UP * 0.15)

        # Single A-plane slice (thin beam going into tissue)
        beam = Line(
            probe.get_bottom(),
            probe.get_bottom() + DOWN * 3.5,
            color=BLUE_B,
            stroke_width=3,
            stroke_opacity=0.6,
        )
        beam_label = Text("single A-plane", font_size=16, color=BLUE_B).next_to(
            beam, LEFT, buff=0.15
        )
        beam_group = VGroup(probe, beam, beam_label)

        self.play(FadeIn(probe), Create(beam), FadeIn(beam_label))
        self.wait(0.5)

        # Good alignment
        check = Text("✓ Slice hits heart", font_size=24, color=GREEN).shift(
            RIGHT * 3.5 + DOWN * 0.5
        )
        self.play(FadeIn(check))
        self.wait(1)

        # Tilt → beam misses
        self.play(FadeOut(check))
        self.play(
            beam_group.animate.shift(RIGHT * 1.8).rotate(
                -15 * DEGREES, about_point=probe.get_top()
            ),
            run_time=1.5,
        )
        cross = Text("✗ Slice misses heart", font_size=24, color=RED).shift(
            RIGHT * 3.5 + DOWN * 0.5
        )
        self.play(FadeIn(cross))
        self.wait(1)

        explanation = Text(
            "2D ultrasound captures ONE slice.\n"
            "Tilt the probe slightly → you miss the anatomy.",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(explanation))
        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 2: 3D Ultrasound Solution (pyramid volume)
    # ------------------------------------------------------------------ #
    def scene2_3d_solution(self):
        title = Text("3D Ultrasound", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))

        # Skin
        skin = Line(LEFT * 5, RIGHT * 5, color=LIGHT_BROWN).shift(UP * 1.5)
        skin_label = Text("skin", font_size=20, color=LIGHT_BROWN).next_to(
            skin, RIGHT, buff=0.2
        )
        self.play(Create(skin), FadeIn(skin_label))

        # Heart
        heart = self._make_heart().scale(0.8).shift(DOWN * 1.2)
        heart_label = Text("heart", font_size=20, color=RED_C).next_to(
            heart, DOWN, buff=0.2
        )
        self.play(FadeIn(heart), FadeIn(heart_label))

        # 3D probe (matrix array) on skin
        probe = Rectangle(width=1.0, height=0.3, color=TEAL, fill_opacity=0.8)
        probe.move_to(skin.get_center() + UP * 0.15)
        dots = VGroup(
            *[
                Dot(radius=0.03, color=WHITE).move_to(
                    probe.get_center()
                    + RIGHT * (c - 1.5) * 0.2
                    + UP * (r - 0.5) * 0.1
                )
                for r in range(2)
                for c in range(4)
            ]
        )
        probe_group = VGroup(probe, dots)

        # Pyramid volume (isometric-ish projection)
        # The probe sits at the apex; the volume fans out downward
        apex_y = probe.get_bottom()[1] - 0.05
        apex_x = probe.get_center()[0]
        apex = np.array([apex_x, apex_y, 0])

        # Base parallelogram (azimuth = right edge, elevation = left edge)
        b_fl = apex + np.array([-2.0, -3.5, 0])
        b_fr = apex + np.array([1.0, -3.5, 0])
        b_br = apex + np.array([2.2, -2.3, 0])
        b_bl = apex + np.array([-0.8, -2.3, 0])

        pyramid = self._make_pyramid(
            apex, b_fl, b_fr, b_br, b_bl, color=TEAL_B, stroke_width=1.5, stroke_opacity=0.5
        )
        # Semi-transparent fill for the front face
        front_face = Polygon(
            apex, b_fl, b_fr,
            color=TEAL, fill_color=TEAL, fill_opacity=0.08, stroke_width=0,
        )
        side_face_r = Polygon(
            apex, b_fr, b_br,
            color=TEAL, fill_color=TEAL, fill_opacity=0.06, stroke_width=0,
        )
        volume_group = VGroup(front_face, side_face_r, pyramid)

        # Beam direction arrow
        beam_arr = Arrow(
            apex + RIGHT * 2.5 + UP * 0.3,
            apex + RIGHT * 2.5 + DOWN * 1.2,
            color=WHITE, stroke_width=2, buff=0,
        )
        beam_lbl = Text("beam\ndirection", font_size=14).next_to(
            beam_arr, RIGHT, buff=0.1
        )

        # Axis labels
        az_label = Text("azimuth", font_size=16, slant=ITALIC, color=TEAL_B).move_to(
            (b_fr + b_br) / 2 + RIGHT * 0.7 + DOWN * 0.2
        )
        el_label = Text("elevation", font_size=16, slant=ITALIC, color=TEAL_B).move_to(
            (b_fl + b_bl) / 2 + LEFT * 0.7 + DOWN * 0.2
        )

        full_group = VGroup(
            probe_group, volume_group, beam_arr, beam_lbl, az_label, el_label
        )

        self.play(FadeIn(probe_group), FadeIn(volume_group))
        self.play(FadeIn(beam_arr), FadeIn(beam_lbl), FadeIn(az_label), FadeIn(el_label))
        self.wait(0.5)

        check = Text("✓ Volume captures heart", font_size=24, color=GREEN).shift(
            LEFT * 3.5 + DOWN * 1.0
        )
        self.play(FadeIn(check))
        self.wait(1)

        # Tilt → heart still inside the volume
        self.play(FadeOut(check))
        self.play(
            full_group.animate.shift(RIGHT * 0.8).rotate(
                -10 * DEGREES, about_point=probe.get_top()
            ),
            run_time=1.5,
        )
        check2 = Text("✓ Still captured!", font_size=24, color=GREEN).shift(
            LEFT * 3.5 + DOWN * 1.0
        )
        self.play(FadeIn(check2))
        self.wait(1)

        explanation = Text(
            "3D ultrasound captures a VOLUME.\n"
            "Even with imperfect orientation, the anatomy is still captured.",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(explanation))
        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 3: A, B, C planes (matching paper Figure 1)
    # ------------------------------------------------------------------ #
    def scene3_reslicing(self):
        title = Text("Reslicing the 3D Volume", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))

        # Isometric pyramid (centered, larger, matching paper fig 1 style)
        apex = np.array([0, 1.5, 0])
        b_fl = np.array([-2.8, -2.2, 0])
        b_fr = np.array([1.0, -2.2, 0])
        b_br = np.array([3.0, -0.5, 0])
        b_bl = np.array([-0.8, -0.5, 0])

        pyramid = self._make_pyramid(
            apex, b_fl, b_fr, b_br, b_bl, color=GRAY, stroke_width=1.5, stroke_opacity=0.6
        )

        # Probe at apex
        probe = Rectangle(width=0.5, height=0.15, color=TEAL, fill_opacity=0.8)
        probe.move_to(apex + UP * 0.12)

        # Beam direction arrow
        beam_arr = Arrow(
            apex + RIGHT * 2.2 + UP * 0.8,
            apex + RIGHT * 2.2 + DOWN * 0.8,
            color=WHITE, stroke_width=2, buff=0,
        )
        beam_lbl = Text("beam direction", font_size=14).next_to(
            beam_arr, RIGHT, buff=0.1
        )

        # Axis labels
        az_label = Text("azimuth", font_size=18, slant=ITALIC, color=GRAY_B).move_to(
            (b_br + b_fr) / 2 + RIGHT * 0.5 + DOWN * 0.4
        )
        el_label = Text("elevation", font_size=18, slant=ITALIC, color=GRAY_B).move_to(
            (b_fl + b_bl) / 2 + LEFT * 0.5 + DOWN * 0.4
        )

        self.play(
            Create(pyramid), FadeIn(probe),
            FadeIn(beam_arr), FadeIn(beam_lbl),
            FadeIn(az_label), FadeIn(el_label),
        )
        self.wait(1)

        # --- A plane (pink): azimuth × depth at fixed elevation ---
        # Slice from apex to a line parallel to azimuth (front/back edges)
        # at ~50% elevation (midpoints of the left and right edges)
        a_left = (b_fl + b_bl) / 2
        a_right = (b_fr + b_br) / 2
        a_plane = Polygon(
            apex, a_left, a_right,
            color="#E8A0BF", fill_color="#E8A0BF", fill_opacity=0.3, stroke_width=2,
        )
        a_label = Text("A", font_size=30, weight=BOLD, color="#E8A0BF").move_to(
            (a_left + a_right) / 2 + DOWN * 0.3
        )
        a_desc = Text(
            "A plane: azimuth × depth  (standard 2D view)", font_size=18, color="#E8A0BF"
        ).to_edge(DOWN, buff=1.2)

        self.play(FadeIn(a_plane, shift=DOWN * 0.3), FadeIn(a_label), FadeIn(a_desc))
        self.wait(1.5)

        # --- B plane (green): elevation × depth at fixed azimuth ---
        frac = 0.42
        b_front_pt = b_fl + frac * (b_fr - b_fl)
        b_back_pt = b_bl + frac * (b_br - b_bl)
        b_plane = Polygon(
            apex, b_front_pt, b_back_pt,
            color=GREEN, fill_color=GREEN, fill_opacity=0.25, stroke_width=2,
        )
        b_label_obj = Text("B", font_size=30, weight=BOLD, color=GREEN).move_to(
            (b_front_pt + b_back_pt) / 2 + LEFT * 0.6
        )
        b_desc = Text(
            "B plane: elevation × depth  (perpendicular view)", font_size=18, color=GREEN
        )
        b_desc.next_to(a_desc, DOWN, buff=0.15)

        self.play(FadeIn(b_plane, shift=RIGHT * 0.3), FadeIn(b_label_obj), FadeIn(b_desc))
        self.wait(1.5)

        # --- C plane (blue): azimuth × elevation at fixed depth ---
        d_frac = 0.5
        c1 = apex + d_frac * (b_fl - apex)
        c2 = apex + d_frac * (b_fr - apex)
        c3 = apex + d_frac * (b_br - apex)
        c4 = apex + d_frac * (b_bl - apex)
        c_plane = Polygon(
            c1, c2, c3, c4,
            color=BLUE, fill_color=BLUE, fill_opacity=0.25, stroke_width=2,
        )
        c_label = Text("C", font_size=30, weight=BOLD, color=BLUE).move_to(
            c3 + RIGHT * 0.4
        )
        c_desc = Text(
            "C plane: azimuth × elevation  (at fixed depth)", font_size=18, color=BLUE
        )
        c_desc.next_to(b_desc, DOWN, buff=0.15)

        self.play(FadeIn(c_plane, shift=UP * 0.3), FadeIn(c_label), FadeIn(c_desc))
        self.wait(1.5)

        explanation = Text(
            "Any desired view can be extracted\ncomputationally after scanning.",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.3)
        self.play(
            FadeOut(a_desc), FadeOut(b_desc), FadeOut(c_desc), FadeIn(explanation)
        )
        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 4: Dense vs sparse elevation planes + AI
    # ------------------------------------------------------------------ #
    def scene4_tradeoff(self):
        title = Text("The Speed–Quality Tradeoff", font_size=36, weight=BOLD).to_edge(UP)
        self.play(Write(title))

        # Each box represents the volume viewed so that elevation planes
        # are visible as vertical lines (elevation × depth cross-section).
        el_axis_label = Text(
            "← elevation →", font_size=16, color=GRAY_B
        ).shift(DOWN * 3.2)
        depth_label = Text(
            "depth ↓", font_size=16, color=GRAY_B
        ).shift(LEFT * 6 + DOWN * 0.3).rotate(90 * DEGREES)
        self.play(FadeIn(el_axis_label), FadeIn(depth_label))

        # --- Dense elevation sampling (left) ---
        dense_box = Rectangle(
            width=2.5, height=2.5, color=TEAL, fill_opacity=0.08, stroke_width=2
        ).shift(LEFT * 3.5 + DOWN * 0.3)
        dense_label = Text(
            "Dense elevation\nsampling", font_size=18, weight=BOLD
        ).next_to(dense_box, UP, buff=0.15)

        dense_lines = VGroup()
        for i in range(10):
            x = dense_box.get_left()[0] + 0.25 + i * 0.22
            dense_lines.add(
                Line(
                    [x, dense_box.get_top()[1] - 0.1, 0],
                    [x, dense_box.get_bottom()[1] + 0.1, 0],
                    color=BLUE_B, stroke_width=2, stroke_opacity=0.7,
                )
            )
        dense_tag = Text("Complete, but slow ✗", font_size=18, color=RED).next_to(
            dense_box, DOWN, buff=0.15
        )

        self.play(Create(dense_box), FadeIn(dense_label))
        self.play(LaggedStartMap(Create, dense_lines, lag_ratio=0.1), run_time=1.5)
        self.play(FadeIn(dense_tag))

        # --- Sparse elevation sampling (center) ---
        sparse_box = Rectangle(
            width=2.5, height=2.5, color=TEAL, fill_opacity=0.08, stroke_width=2
        ).shift(DOWN * 0.3)
        sparse_label = Text(
            "Sparse elevation\nsampling", font_size=18, weight=BOLD
        ).next_to(sparse_box, UP, buff=0.15)

        sparse_lines = VGroup()
        for i in range(3):
            x = sparse_box.get_left()[0] + 0.5 + i * 0.75
            sparse_lines.add(
                Line(
                    [x, sparse_box.get_top()[1] - 0.1, 0],
                    [x, sparse_box.get_bottom()[1] + 0.1, 0],
                    color=BLUE_B, stroke_width=2, stroke_opacity=0.7,
                )
            )
        sparse_tag = Text("Fast ✓  but gaps ✗", font_size=18, color=GOLD).next_to(
            sparse_box, DOWN, buff=0.15
        )

        self.play(Create(sparse_box), FadeIn(sparse_label))
        self.play(LaggedStartMap(Create, sparse_lines, lag_ratio=0.15), run_time=1)
        self.play(FadeIn(sparse_tag))
        self.wait(1)

        # --- AI reconstruction (right) ---
        ai_box = Rectangle(
            width=2.5, height=2.5, color=GREEN, fill_opacity=0.08, stroke_width=2
        ).shift(RIGHT * 3.5 + DOWN * 0.3)
        ai_label = Text(
            "AI reconstruction", font_size=18, weight=BOLD, color=GREEN
        ).next_to(ai_box, UP, buff=0.15)

        positions = [ai_box.get_left()[0] + 0.25 + i * 0.22 for i in range(10)]
        real_idx = {0, 4, 9}
        ai_real = VGroup()
        ai_gen = VGroup()
        for i, x in enumerate(positions):
            line = Line(
                [x, ai_box.get_top()[1] - 0.1, 0],
                [x, ai_box.get_bottom()[1] + 0.1, 0],
                stroke_width=2,
                stroke_opacity=0.7,
                color=BLUE_B if i in real_idx else GREEN_B,
            )
            (ai_real if i in real_idx else ai_gen).add(line)

        ai_tag = Text("Fast ✓  + complete ✓", font_size=18, color=GREEN).next_to(
            ai_box, DOWN, buff=0.15
        )

        self.play(Create(ai_box), FadeIn(ai_label))
        self.play(LaggedStartMap(Create, ai_real, lag_ratio=0.1), run_time=0.5)
        self.play(LaggedStartMap(Create, ai_gen, lag_ratio=0.05), run_time=1.5)
        self.play(FadeIn(ai_tag))

        # Legend for AI box
        legend = VGroup(
            VGroup(
                Line(ORIGIN, RIGHT * 0.4, color=BLUE_B, stroke_width=3),
                Text("acquired", font_size=14, color=BLUE_B),
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Line(ORIGIN, RIGHT * 0.4, color=GREEN_B, stroke_width=3),
                Text("AI-generated", font_size=14, color=GREEN_B),
            ).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.next_to(ai_box, RIGHT, buff=0.15).shift(UP * 0.3)
        self.play(FadeIn(legend))

        explanation = Text(
            "This paper uses AI to fill the gaps\n"
            "between sparsely sampled elevation planes.",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(explanation))
        self.wait(4)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _make_heart(self):
        """Return a simple heart-shaped VMobject."""
        t_values = np.linspace(0, 2 * np.pi, 100)
        points = []
        for t in t_values:
            x = 16 * np.sin(t) ** 3
            y = (
                13 * np.cos(t)
                - 5 * np.cos(2 * t)
                - 2 * np.cos(3 * t)
                - np.cos(4 * t)
            )
            points.append([x / 17, y / 17, 0])
        heart = VMobject(
            color=RED_C, fill_color=RED_E, fill_opacity=0.4, stroke_width=2
        )
        heart.set_points_smoothly([np.array(p) for p in points])
        return heart

    def _make_pyramid(self, apex, b_fl, b_fr, b_br, b_bl, **kw):
        """Return wireframe edges of a pyramid with a parallelogram base."""
        c = kw.get("color", GRAY)
        sw = kw.get("stroke_width", 2)
        so = kw.get("stroke_opacity", 1)
        mk = dict(color=c, stroke_width=sw, stroke_opacity=so)
        return VGroup(
            # Apex to base corners
            Line(apex, b_fl, **mk),
            Line(apex, b_fr, **mk),
            Line(apex, b_br, **mk),
            Line(apex, b_bl, **mk),
            # Base edges
            Line(b_fl, b_fr, **mk),
            Line(b_fr, b_br, **mk),
            Line(b_br, b_bl, **mk),
            Line(b_bl, b_fl, **mk),
        )
