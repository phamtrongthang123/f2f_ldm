"""
Manim animation: Why diverging waves degrade 3D ultrasound image quality.

Two problems:
  1. Limited tissue harmonic generation (energy spread → low pressure → weak harmonics)
  2. Multipath effects (sound bounces off multiple structures → ghost artifacts)

Render with:
    manim -qm substack/harmonics.py DivergingWaveProblems
"""

from manim import *
import numpy as np


class DivergingWaveProblems(Scene):
    def construct(self):
        self.scene1_focused_vs_diverging()
        self.clear_scene()
        self.scene2_pressure_and_harmonics()
        self.clear_scene()
        self.scene3_multipath_focused()
        self.clear_scene()
        self.scene4_multipath_diverging()
        self.clear_scene()
        self.scene5_summary()

    def clear_scene(self):
        self.play(FadeOut(*self.mobjects), run_time=0.5)
        self.wait(0.3)

    # ------------------------------------------------------------------ #
    # Scene 1: Show what focused vs diverging beams look like
    # ------------------------------------------------------------------ #
    def scene1_focused_vs_diverging(self):
        title = Text("Focused Beam vs Diverging Wave", font_size=30, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # --- Left: focused beam ---
        left_label = Text("Focused", font_size=24, color=GREEN, weight=BOLD)
        left_label.shift(LEFT * 3.5 + UP * 2)
        self.play(FadeIn(left_label))

        # Probe
        probe_l = Rectangle(width=1.8, height=0.3, color=BLUE, fill_opacity=0.8)
        probe_l.shift(LEFT * 3.5 + UP * 1.5)
        probe_l_label = Text("probe", font_size=16, color=BLUE_B)
        probe_l_label.next_to(probe_l, RIGHT, buff=0.15)
        self.play(FadeIn(probe_l), FadeIn(probe_l_label))

        # Focused beam: converging lines → focal point → diverge slightly
        focal_y = -0.5
        beam_l_lines = VGroup()
        for dx in [-0.6, -0.3, 0, 0.3, 0.6]:
            line = Line(
                probe_l.get_bottom() + RIGHT * dx,
                np.array([-3.5, focal_y, 0]),
                color=GREEN, stroke_width=2, stroke_opacity=0.7,
            )
            beam_l_lines.add(line)
        # Past focal point
        for dx in [-0.3, 0, 0.3]:
            line = Line(
                np.array([-3.5, focal_y, 0]),
                np.array([-3.5 + dx * 0.8, -2.5, 0]),
                color=GREEN, stroke_width=1.5, stroke_opacity=0.4,
            )
            beam_l_lines.add(line)

        focal_dot = Dot(np.array([-3.5, focal_y, 0]), color=YELLOW, radius=0.08)
        focal_label = Text("focal point\n(high pressure)", font_size=14, color=YELLOW)
        focal_label.next_to(focal_dot, RIGHT, buff=0.2)

        self.play(Create(beam_l_lines), run_time=1.5)
        self.play(FadeIn(focal_dot), FadeIn(focal_label))

        # --- Right: diverging wave ---
        right_label = Text("Diverging", font_size=24, color=RED_C, weight=BOLD)
        right_label.shift(RIGHT * 3.5 + UP * 2)
        self.play(FadeIn(right_label))

        probe_r = Rectangle(width=1.8, height=0.3, color=BLUE, fill_opacity=0.8)
        probe_r.shift(RIGHT * 3.5 + UP * 1.5)
        probe_r_label = Text("probe", font_size=16, color=BLUE_B)
        probe_r_label.next_to(probe_r, RIGHT, buff=0.15)
        self.play(FadeIn(probe_r), FadeIn(probe_r_label))

        # Diverging beam: lines fan out
        beam_r_lines = VGroup()
        for dx, spread in [(-0.4, -1.8), (-0.2, -0.9), (0, 0), (0.2, 0.9), (0.4, 1.8)]:
            line = Line(
                probe_r.get_bottom() + RIGHT * dx,
                np.array([3.5 + spread, -2.5, 0]),
                color=RED_C, stroke_width=2, stroke_opacity=0.7,
            )
            beam_r_lines.add(line)

        spread_label = Text("energy spreads out\n(low pressure everywhere)", font_size=14, color=RED_B)
        spread_label.shift(RIGHT * 3.5 + DOWN * 1.5)

        self.play(Create(beam_r_lines), run_time=1.5)
        self.play(FadeIn(spread_label))

        # Separator
        sep = DashedLine(UP * 2.5, DOWN * 3, color=GREY, stroke_width=1)
        self.play(Create(sep))

        # Bottom note
        note = Text(
            "Diverging wave covers a wide area in 1 transmit (fast!), but with less energy per point",
            font_size=18, color=GREY_B,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 2: Problem 1 — Limited harmonic generation
    # ------------------------------------------------------------------ #
    def scene2_pressure_and_harmonics(self):
        title = Text("Problem 1: Limited Tissue Harmonic Generation", font_size=28, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Explain in two rows: focused (top) and diverging (bottom)

        # --- Top row: Focused, high pressure ---
        row1_label = Text("Focused beam → high pressure", font_size=20, color=GREEN)
        row1_label.shift(LEFT * 3.5 + UP * 2)
        self.play(FadeIn(row1_label))

        ax1 = Axes(
            x_range=[0, 4 * PI, PI], y_range=[-1.5, 1.5, 0.5],
            x_length=4.5, y_length=1.8,
            axis_config={"include_tip": False, "stroke_width": 1},
        ).shift(LEFT * 3.5 + UP * 0.5)

        # High pressure → strong distortion
        wave_focused = ax1.plot(
            lambda x: np.sin(x) - 0.35 * np.sin(2 * x) + 0.15 * np.sin(3 * x),
            color=GREEN, x_range=[0, 4 * PI],
        )
        label_focused = Text("strongly distorted", font_size=16, color=GREEN)
        label_focused.next_to(ax1, RIGHT, buff=0.2)

        self.play(Create(ax1), Create(wave_focused), FadeIn(label_focused), run_time=1)

        # Arrow showing harmonics
        arrow1 = Arrow(LEFT * 0.8 + UP * 0.5, RIGHT * 0.8 + UP * 0.5, color=GREEN, buff=0)
        arrow1_label = Text("strong\nharmonics", font_size=14, color=GREEN)
        arrow1_label.next_to(arrow1, UP, buff=0.1)
        self.play(Create(arrow1), FadeIn(arrow1_label))

        # Mini frequency spectrum for focused
        ax1_freq = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.2, 0.5],
            x_length=3, y_length=1.5,
            axis_config={"include_tip": False, "stroke_width": 1},
        ).shift(RIGHT * 3.5 + UP * 0.5)

        bars_focused = VGroup(
            Rectangle(width=0.4, height=1.5, fill_opacity=0.8, color=GREEN).move_to(
                ax1_freq.c2p(1, 0.75)),
            Rectangle(width=0.4, height=0.8, fill_opacity=0.8, color=GREEN_B).move_to(
                ax1_freq.c2p(2, 0.4)),
            Rectangle(width=0.4, height=0.35, fill_opacity=0.8, color=GREEN_A).move_to(
                ax1_freq.c2p(3, 0.175)),
        )
        freq_labels_top = VGroup(
            Text("f", font_size=14).next_to(ax1_freq.c2p(1, 0), DOWN, buff=0.1),
            Text("2f", font_size=14).next_to(ax1_freq.c2p(2, 0), DOWN, buff=0.1),
            Text("3f", font_size=14).next_to(ax1_freq.c2p(3, 0), DOWN, buff=0.1),
        )

        self.play(Create(ax1_freq), FadeIn(bars_focused), FadeIn(freq_labels_top), run_time=1)

        # --- Bottom row: Diverging, low pressure ---
        row2_label = Text("Diverging wave → low pressure", font_size=20, color=RED_C)
        row2_label.shift(LEFT * 3.5 + DOWN * 1.2)
        self.play(FadeIn(row2_label))

        ax2 = Axes(
            x_range=[0, 4 * PI, PI], y_range=[-1.5, 1.5, 0.5],
            x_length=4.5, y_length=1.8,
            axis_config={"include_tip": False, "stroke_width": 1},
        ).shift(LEFT * 3.5 + DOWN * 2.5)

        # Low pressure → almost no distortion (nearly pure sine, just smaller)
        wave_diverging = ax2.plot(
            lambda x: 0.4 * np.sin(x) - 0.02 * np.sin(2 * x),
            color=RED_C, x_range=[0, 4 * PI],
        )
        label_diverging = Text("barely distorted", font_size=16, color=RED_C)
        label_diverging.next_to(ax2, RIGHT, buff=0.2)

        self.play(Create(ax2), Create(wave_diverging), FadeIn(label_diverging), run_time=1)

        # Arrow
        arrow2 = Arrow(LEFT * 0.8 + DOWN * 2.5, RIGHT * 0.8 + DOWN * 2.5, color=RED_C, buff=0)
        arrow2_label = Text("weak\nharmonics", font_size=14, color=RED_C)
        arrow2_label.next_to(arrow2, UP, buff=0.1)
        self.play(Create(arrow2), FadeIn(arrow2_label))

        # Mini frequency spectrum for diverging
        ax2_freq = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.2, 0.5],
            x_length=3, y_length=1.5,
            axis_config={"include_tip": False, "stroke_width": 1},
        ).shift(RIGHT * 3.5 + DOWN * 2.5)

        bars_diverging = VGroup(
            Rectangle(width=0.4, height=0.6, fill_opacity=0.8, color=RED_C).move_to(
                ax2_freq.c2p(1, 0.3)),
            Rectangle(width=0.4, height=0.06, fill_opacity=0.8, color=RED_B).move_to(
                ax2_freq.c2p(2, 0.03)),
            Rectangle(width=0.4, height=0.01, fill_opacity=0.8, color=RED_A).move_to(
                ax2_freq.c2p(3, 0.005)),
        )
        freq_labels_bot = VGroup(
            Text("f", font_size=14).next_to(ax2_freq.c2p(1, 0), DOWN, buff=0.1),
            Text("2f", font_size=14).next_to(ax2_freq.c2p(2, 0), DOWN, buff=0.1),
            Text("3f", font_size=14).next_to(ax2_freq.c2p(3, 0), DOWN, buff=0.1),
        )

        self.play(Create(ax2_freq), FadeIn(bars_diverging), FadeIn(freq_labels_bot), run_time=1)

        # Bottom note
        note = Text(
            "Without harmonics, the probe must use the noisy fundamental signal → worse image",
            font_size=18, color=YELLOW,
        ).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(note))
        self.wait(4)

    # ------------------------------------------------------------------ #
    # Scene 3: Multipath — focused beam (clean)
    # ------------------------------------------------------------------ #
    def scene3_multipath_focused(self):
        title = Text("Problem 2: Multipath Effects", font_size=28, weight=BOLD)
        title.to_edge(UP)
        subtitle = Text("Focused beam — clean echoes", font_size=22, color=GREEN)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle))

        # Probe at top
        probe = Rectangle(width=2, height=0.3, color=BLUE, fill_opacity=0.8)
        probe.shift(UP * 2)
        self.play(FadeIn(probe))

        # Tissue structures (two ellipses)
        struct_a = Ellipse(width=1.2, height=0.6, color=LIGHT_BROWN, fill_opacity=0.3)
        struct_a.shift(DOWN * 0.5)
        label_a = Text("A", font_size=20, color=LIGHT_BROWN).move_to(struct_a)

        struct_b = Ellipse(width=1.0, height=0.5, color=LIGHT_BROWN, fill_opacity=0.3)
        struct_b.shift(RIGHT * 2.5 + DOWN * 1.5)
        label_b = Text("B", font_size=20, color=LIGHT_BROWN).move_to(struct_b)

        self.play(FadeIn(struct_a), FadeIn(label_a), FadeIn(struct_b), FadeIn(label_b))

        # Focused beam: narrow, only hits structure A
        beam_down = Arrow(
            probe.get_bottom(), struct_a.get_top(),
            color=GREEN, stroke_width=3, buff=0.05,
        )
        beam_label = Text("narrow beam", font_size=16, color=GREEN)
        beam_label.next_to(beam_down, LEFT, buff=0.2)
        self.play(Create(beam_down), FadeIn(beam_label), run_time=1)
        self.wait(0.5)

        # Echo comes straight back
        echo_up = Arrow(
            struct_a.get_top() + RIGHT * 0.15,
            probe.get_bottom() + RIGHT * 0.15,
            color=YELLOW, stroke_width=3, buff=0.05,
        )
        echo_label = Text("clean echo\nfrom A only", font_size=16, color=YELLOW)
        echo_label.next_to(echo_up, RIGHT, buff=0.2)
        self.play(Create(echo_up), FadeIn(echo_label), run_time=1)

        # Result
        result = Text(
            "Probe knows exactly where the echo came from",
            font_size=20, color=GREEN,
        ).to_edge(DOWN, buff=0.5)
        check = Text("clean image", font_size=22, color=GREEN, weight=BOLD)
        check.next_to(result, DOWN, buff=0.2)
        self.play(FadeIn(result), FadeIn(check))
        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 4: Multipath — diverging wave (messy)
    # ------------------------------------------------------------------ #
    def scene4_multipath_diverging(self):
        title = Text("Problem 2: Multipath Effects", font_size=28, weight=BOLD)
        title.to_edge(UP)
        subtitle = Text("Diverging wave — confusing echoes", font_size=22, color=RED_C)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle))

        # Probe at top
        probe = Rectangle(width=2, height=0.3, color=BLUE, fill_opacity=0.8)
        probe.shift(UP * 2)
        self.play(FadeIn(probe))

        # Same tissue structures
        struct_a = Ellipse(width=1.2, height=0.6, color=LIGHT_BROWN, fill_opacity=0.3)
        struct_a.shift(DOWN * 0.5)
        label_a = Text("A", font_size=20, color=LIGHT_BROWN).move_to(struct_a)

        struct_b = Ellipse(width=1.0, height=0.5, color=LIGHT_BROWN, fill_opacity=0.3)
        struct_b.shift(RIGHT * 2.5 + DOWN * 1.5)
        label_b = Text("B", font_size=20, color=LIGHT_BROWN).move_to(struct_b)

        self.play(FadeIn(struct_a), FadeIn(label_a), FadeIn(struct_b), FadeIn(label_b))

        # Diverging beam: wide, hits everything
        beam_lines = VGroup()
        for target in [LEFT * 2.5 + DOWN * 2.5, LEFT * 1 + DOWN * 2.5,
                        struct_a.get_top(),
                        struct_b.get_top(),
                        RIGHT * 3.5 + DOWN * 2.5]:
            line = Arrow(
                probe.get_bottom(), target,
                color=RED_C, stroke_width=2, stroke_opacity=0.6, buff=0.05,
            )
            beam_lines.add(line)
        beam_label = Text("sound goes\neverywhere", font_size=16, color=RED_C)
        beam_label.shift(LEFT * 3.5 + DOWN * 0.5)
        self.play(Create(beam_lines), FadeIn(beam_label), run_time=1.5)
        self.wait(0.5)

        # Direct echo from A (good)
        echo_direct = Arrow(
            struct_a.get_top() + LEFT * 0.15,
            probe.get_bottom() + LEFT * 0.3,
            color=YELLOW, stroke_width=2, buff=0.05,
        )
        echo_direct_label = Text("direct echo\nfrom A", font_size=14, color=YELLOW)
        echo_direct_label.next_to(echo_direct, LEFT, buff=0.15)
        self.play(Create(echo_direct), FadeIn(echo_direct_label), run_time=0.8)

        # Multipath echo: sound hits B, bounces to A, then back to probe
        bounce1 = Arrow(
            struct_b.get_top(), struct_a.get_bottom() + RIGHT * 0.3,
            color=ORANGE, stroke_width=2.5, buff=0.05,
        )
        bounce2 = Arrow(
            struct_a.get_top() + RIGHT * 0.2,
            probe.get_bottom() + RIGHT * 0.3,
            color=ORANGE, stroke_width=2.5, buff=0.05,
        )

        bounce_label = Text("multipath:\nhits B → bounces to A → probe", font_size=14, color=ORANGE)
        bounce_label.shift(RIGHT * 2 + UP * 0.5)

        self.play(Create(bounce1), run_time=0.8)
        self.play(Create(bounce2), FadeIn(bounce_label), run_time=0.8)

        # Show the problem: longer path = appears deeper
        problem_box = VGroup(
            Text("This echo traveled a longer path", font_size=18, color=ORANGE),
            Text("→ arrives later than the direct echo", font_size=18, color=ORANGE),
            Text("→ probe thinks it came from deeper", font_size=18, color=ORANGE),
            Text("→ GHOST artifact in the image!", font_size=20, color=RED, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(DOWN, buff=0.3)

        for line in problem_box:
            self.play(FadeIn(line), run_time=0.6)

        self.wait(3)

    # ------------------------------------------------------------------ #
    # Scene 5: Summary
    # ------------------------------------------------------------------ #
    def scene5_summary(self):
        title = Text("Why Diverging Waves Degrade Image Quality", font_size=28, weight=BOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Two boxes side by side
        box1_title = Text("1. Limited Harmonics", font_size=22, color=BLUE, weight=BOLD)
        box1_content = VGroup(
            Text("Energy spreads out", font_size=18),
            Text("→ Low pressure at each point", font_size=18),
            Text("→ Tissue barely distorts the wave", font_size=18),
            Text("→ No harmonic signal to image with", font_size=18),
            Text("→ Noisier, lower-quality image", font_size=18, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        box1 = VGroup(box1_title, box1_content).arrange(DOWN, buff=0.3)
        box1_rect = SurroundingRectangle(box1, color=BLUE, buff=0.25, corner_radius=0.1)
        box1_group = VGroup(box1_rect, box1)

        box2_title = Text("2. Multipath Effects", font_size=22, color=ORANGE, weight=BOLD)
        box2_content = VGroup(
            Text("Sound goes in all directions", font_size=18),
            Text("→ Bounces off multiple structures", font_size=18),
            Text("→ Echoes arrive with wrong timing", font_size=18),
            Text("→ Probe misinterprets their origin", font_size=18),
            Text("→ Ghost artifacts in the image", font_size=18, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        box2 = VGroup(box2_title, box2_content).arrange(DOWN, buff=0.3)
        box2_rect = SurroundingRectangle(box2, color=ORANGE, buff=0.25, corner_radius=0.1)
        box2_group = VGroup(box2_rect, box2)

        both = VGroup(box1_group, box2_group).arrange(RIGHT, buff=0.6).shift(DOWN * 0.3)

        self.play(FadeIn(box1_group), run_time=1)
        self.wait(1)
        self.play(FadeIn(box2_group), run_time=1)
        self.wait(1)

        # Bottom: the tradeoff
        tradeoff = Text(
            "Tradeoff: diverging waves are FAST but produce WORSE images",
            font_size=22, color=YELLOW, weight=BOLD,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(tradeoff))
        self.wait(4)
