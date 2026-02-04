from manim import *
import numpy as np


class UltrasoundAcquisition(Scene):
    def construct(self):
        self.part1_focused_beams()
        self.clear_scene()
        self.part2_diverging_waves()
        self.clear_scene()
        self.part3_elevation_planes()
        self.clear_scene()
        self.part4_interpolation()

    def clear_scene(self):
        self.play(FadeOut(*self.mobjects), run_time=0.5)
        self.wait(0.3)

    # ------------------------------------------------------------------ #
    #  Part 1: Traditional Focused Beams
    # ------------------------------------------------------------------ #
    def part1_focused_beams(self):
        title = Text("Part 1: Focused Beams in Azimuth", font_size=32).to_edge(UP)
        self.play(Write(title), run_time=0.8)

        # Draw the elevation plane as a rectangle (azimuth x depth)
        plane_w, plane_h = 8, 4
        plane = Rectangle(
            width=plane_w, height=plane_h,
            stroke_color=WHITE, stroke_width=2
        ).shift(DOWN * 0.3)
        az_label = Text("Azimuth →", font_size=20).next_to(plane, DOWN, buff=0.15)
        depth_label = Text("← Depth", font_size=20).rotate(PI / 2).next_to(plane, LEFT, buff=0.15)
        self.play(Create(plane), Write(az_label), Write(depth_label), run_time=0.8)

        # Transmission counter
        counter_label = Text("Transmissions: ", font_size=24).to_corner(UR).shift(LEFT * 1.5)
        counter_num = Integer(0, font_size=28).next_to(counter_label, RIGHT)
        self.play(Write(counter_label), Write(counter_num), run_time=0.5)

        # Fire narrow focused beams across azimuth
        n_beams = 16
        beam_positions = np.linspace(
            plane.get_left()[0] + 0.25,
            plane.get_right()[0] - 0.25,
            n_beams,
        )
        top_y = plane.get_top()[1]
        bot_y = plane.get_bottom()[1]

        beams_group = VGroup()
        for i, x in enumerate(beam_positions):
            beam = Line(
                start=[x, top_y, 0],
                end=[x, bot_y, 0],
                stroke_width=3,
                stroke_color=YELLOW,
                stroke_opacity=0.8,
            )
            beams_group.add(beam)
            new_count = Integer(i + 1, font_size=28).move_to(counter_num)
            self.play(
                Create(beam, run_time=0.15),
                counter_num.animate.become(new_count),
                run_time=0.15,
            )
        # Update counter to final value
        final_count = Integer(n_beams, font_size=28).move_to(counter_num)
        counter_num.become(final_count)

        bottom_label = Text(
            "Focused beams in azimuth — many transmissions per plane",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN)
        self.play(Write(bottom_label), run_time=0.8)
        self.wait(1.5)

    # ------------------------------------------------------------------ #
    #  Part 2: Diverging Waves in Azimuth
    # ------------------------------------------------------------------ #
    def part2_diverging_waves(self):
        title = Text("Part 2: Diverging Wave in Azimuth", font_size=32).to_edge(UP)
        self.play(Write(title), run_time=0.8)

        # Draw elevation plane
        plane_w, plane_h = 8, 4
        plane = Rectangle(
            width=plane_w, height=plane_h,
            stroke_color=WHITE, stroke_width=2
        ).shift(DOWN * 0.3)
        az_label = Text("Azimuth →", font_size=20).next_to(plane, DOWN, buff=0.15)
        depth_label = Text("← Depth", font_size=20).rotate(PI / 2).next_to(plane, LEFT, buff=0.15)
        self.play(Create(plane), Write(az_label), Write(depth_label), run_time=0.8)

        # Counter
        counter_label = Text("Transmissions: ", font_size=24).to_corner(UR).shift(LEFT * 1.5)
        counter_num = Integer(0, font_size=28).next_to(counter_label, RIGHT)
        self.play(Write(counter_label), Write(counter_num), run_time=0.5)

        # Diverging wave: a filled sector + expanding wavefront arcs
        source_point = plane.get_top()
        half_angle = PI / 3  # opening half-angle of the diverging wave

        # Filled sector showing the coverage area of ONE transmission
        # Build as a polygon: source -> arc along bottom
        n_pts = 40
        sector_pts = [source_point]
        for i in range(n_pts + 1):
            angle = -PI / 2 - half_angle + (2 * half_angle) * i / n_pts
            x = source_point[0] + plane_h * np.cos(angle)
            y = source_point[1] + plane_h * np.sin(angle)
            # Clamp to plane boundaries
            x = np.clip(x, plane.get_left()[0], plane.get_right()[0])
            y = np.clip(y, plane.get_bottom()[1], plane.get_top()[1])
            sector_pts.append([x, y, 0])
        sector_pts.append(source_point)

        sector = Polygon(
            *sector_pts,
            fill_color=BLUE,
            fill_opacity=0.2,
            stroke_width=0,
        )

        # Two boundary lines for the sector edges
        left_edge = Line(
            source_point,
            [source_point[0] + plane_h * np.cos(-PI / 2 - half_angle),
             max(source_point[1] + plane_h * np.sin(-PI / 2 - half_angle), plane.get_bottom()[1]), 0],
            stroke_color=BLUE, stroke_width=2, stroke_opacity=0.6,
        )
        right_edge = Line(
            source_point,
            [source_point[0] + plane_h * np.cos(-PI / 2 + half_angle),
             max(source_point[1] + plane_h * np.sin(-PI / 2 + half_angle), plane.get_bottom()[1]), 0],
            stroke_color=BLUE, stroke_width=2, stroke_opacity=0.6,
        )

        # Animate: single transmission fires — sector appears all at once
        new_count = Integer(1, font_size=28).move_to(counter_num)
        self.play(
            FadeIn(sector),
            Create(left_edge), Create(right_edge),
            counter_num.animate.become(new_count),
            run_time=0.8,
        )

        # Expanding wavefront arcs to show the wave propagating
        for frac in [0.25, 0.5, 0.75, 1.0]:
            radius = frac * plane_h
            arc = Arc(
                radius=radius,
                start_angle=-PI / 2 - half_angle,
                angle=2 * half_angle,
                stroke_color=BLUE,
                stroke_width=3,
            ).move_arc_center_to(source_point)
            self.play(Create(arc), run_time=0.3)
            self.play(arc.animate.set_stroke(opacity=0.3), run_time=0.15)

        bottom_label = Text(
            "Diverging wave in azimuth — 1 transmission covers entire plane",
            font_size=22, color=BLUE,
        ).to_edge(DOWN)
        self.play(Write(bottom_label), run_time=0.8)

        # Side annotation
        side_note = Text(
            "Still focused in\nelevation (high\nresolution there)",
            font_size=18, color=GREEN,
        ).to_edge(RIGHT).shift(DOWN * 0.5)
        box = SurroundingRectangle(side_note, color=GREEN, buff=0.15, stroke_width=1)
        self.play(Write(side_note), Create(box), run_time=0.8)
        self.wait(1.5)

    # ------------------------------------------------------------------ #
    #  Part 3: Elevation Planes — Full vs Sparse
    # ------------------------------------------------------------------ #
    def part3_elevation_planes(self):
        title = Text("Part 3: Full vs Sparse Elevation Sampling", font_size=32).to_edge(UP)
        self.play(Write(title), run_time=0.8)

        n_planes = 12
        spacing = 0.45
        plane_w, plane_h = 4, 2.5

        def make_plane_stack(x_offset, planes_active, label_text, color, dashed_indices=None):
            """Create a stack of parallelogram planes in pseudo-3D."""
            group = VGroup()
            skew = 0.3  # horizontal skew for 3D look
            for i in range(n_planes):
                y_shift = i * spacing - (n_planes * spacing / 2)
                # Parallelogram corners
                bl = [x_offset - plane_w / 2 + skew * (i / n_planes), y_shift - plane_h / 2 + i * 0.08, 0]
                br = [x_offset + plane_w / 2 + skew * (i / n_planes), y_shift - plane_h / 2 + i * 0.08, 0]
                tr = [x_offset + plane_w / 2 + skew * ((i + 1) / n_planes), y_shift + plane_h / 2 + i * 0.08, 0]
                tl = [x_offset - plane_w / 2 + skew * ((i + 1) / n_planes), y_shift + plane_h / 2 + i * 0.08, 0]
                # Simplify: just use rectangles shifted
                rect = Rectangle(
                    width=plane_w, height=0.02,
                    fill_color=color, fill_opacity=0.7,
                    stroke_color=color, stroke_width=1.5,
                ).shift(RIGHT * x_offset + UP * (y_shift * 0.5) + RIGHT * (i * 0.15))

                if dashed_indices and i in dashed_indices:
                    rect.set_stroke(color, width=1, opacity=0.3)
                    rect.set_fill(color, opacity=0.1)

                if i in planes_active:
                    rect.set_fill(color, opacity=0.7)
                    rect.set_stroke(color, width=2)

                group.add(rect)

            label = Text(label_text, font_size=18).next_to(group, DOWN, buff=0.3)
            return VGroup(group, label)

        # Full acquisition — all planes
        full_planes = list(range(n_planes))
        full_stack = make_plane_stack(-3.5, full_planes, "Full acquisition\n(all planes)", YELLOW)
        self.play(FadeIn(full_stack[1]), run_time=0.5)

        # Animate planes appearing one by one
        for rect in full_stack[0]:
            self.play(FadeIn(rect), run_time=0.12)

        self.wait(0.5)

        # Sparse acquisition — every 3rd plane
        r = 3
        sparse_active = list(range(0, n_planes, r))
        sparse_dashed = [i for i in range(n_planes) if i not in sparse_active]
        sparse_stack = make_plane_stack(3.0, sparse_active, f"Sparse acquisition\n(every {r}rd plane)", BLUE, sparse_dashed)
        self.play(FadeIn(sparse_stack[1]), run_time=0.5)

        for i, rect in enumerate(sparse_stack[0]):
            if i in sparse_active:
                self.play(FadeIn(rect), run_time=0.15)
            else:
                self.play(FadeIn(rect), run_time=0.05)

        bottom_label = Text(
            "Undersampling elevation planes to match diverging wave volume rates",
            font_size=20, color=TEAL,
        ).to_edge(DOWN)
        self.play(Write(bottom_label), run_time=0.8)
        self.wait(1.5)

    # ------------------------------------------------------------------ #
    #  Part 4: Interpolation
    # ------------------------------------------------------------------ #
    def part4_interpolation(self):
        title = Text("Part 4: Interpolation Fills the Gaps", font_size=32).to_edge(UP)
        self.play(Write(title), run_time=0.8)

        n_planes = 12
        r = 3
        spacing = 0.4
        x_center = 0

        planes = VGroup()
        acquired_indices = set(range(0, n_planes, r))

        for i in range(n_planes):
            y_pos = (i - n_planes / 2) * spacing
            rect = Rectangle(
                width=5, height=0.04,
                stroke_width=1.5,
            ).shift(UP * y_pos + RIGHT * (i * 0.12))

            if i in acquired_indices:
                rect.set_fill(BLUE, opacity=0.8)
                rect.set_stroke(BLUE, width=2)
            else:
                rect.set_fill(DARK_GRAY, opacity=0.1)
                rect.set_stroke(GRAY, width=0.5, opacity=0.3)

            planes.add(rect)

        # Show acquired planes first
        acquired_rects = VGroup(*[planes[i] for i in acquired_indices])
        self.play(FadeIn(acquired_rects), run_time=0.8)

        # Label
        acq_label = Text("Acquired planes (solid)", font_size=18, color=BLUE).to_edge(LEFT).shift(DOWN * 2)
        self.play(Write(acq_label), run_time=0.5)

        self.wait(0.5)

        # Animate interpolated planes fading in
        interp_indices = [i for i in range(n_planes) if i not in acquired_indices]
        interp_rects = VGroup()
        for i in interp_indices:
            rect = planes[i].copy()
            rect.set_fill(ORANGE, opacity=0.35)
            rect.set_stroke(ORANGE, width=1, opacity=0.5)
            interp_rects.add(rect)

        interp_label = Text("Interpolated planes (faded)", font_size=18, color=ORANGE).to_edge(RIGHT).shift(DOWN * 2)
        self.play(
            FadeIn(interp_rects, lag_ratio=0.15),
            Write(interp_label),
            run_time=1.5,
        )

        bottom_label = Text(
            "Simple interpolation fills gaps — but introduces artifacts",
            font_size=22, color=ORANGE,
        ).to_edge(DOWN)
        self.play(Write(bottom_label), run_time=0.8)
        self.wait(2)
