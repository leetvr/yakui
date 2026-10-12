use yakui_core::{
    geometry::{Constraints, Rect, Vec2},
    paint::{PaintMesh, PaintTransform, Pipeline, Vertex},
    widget::{LayoutContext, PaintContext, Widget},
    Yakui,
};

#[derive(Debug, Default)]
struct Scope {
    transform: PaintTransform,
    opacity: f32,
}
impl Widget for Scope {
    type Props<'a> = (PaintTransform, f32);
    type Response = ();
    fn new() -> Self {
        Self::default()
    }
    fn update(&mut self, props: Self::Props<'_>) {
        self.transform = props.0;
        self.opacity = props.1;
    }
    fn layout(&self, ctx: LayoutContext<'_>, constraints: Constraints) -> Vec2 {
        constraints.constrain(self.default_layout(ctx, Constraints::loose(constraints.max)))
    }
    fn paint(&self, ctx: PaintContext<'_>) {
        ctx.paint.with_transform(self.transform, |paint| {
            paint.with_opacity(self.opacity, |paint| {
                self.default_paint(PaintContext {
                    dom: ctx.dom,
                    layout: ctx.layout,
                    paint,
                })
            });
        });
    }
}
#[derive(Debug)]
struct Clip;
impl Widget for Clip {
    type Props<'a> = ();
    type Response = ();
    fn new() -> Self {
        Self
    }
    fn update(&mut self, _: ()) {}
    fn layout(&self, ctx: LayoutContext<'_>, constraints: Constraints) -> Vec2 {
        ctx.layout.enable_clipping(ctx.dom);
        self.default_layout(ctx, constraints)
    }
}
#[derive(Debug, Default)]
struct Point {
    position: Vec2,
}
impl Widget for Point {
    type Props<'a> = Vec2;
    type Response = ();
    fn new() -> Self {
        Self::default()
    }
    fn update(&mut self, position: Vec2) {
        self.position = position;
    }
    fn layout(&self, _: LayoutContext<'_>, constraints: Constraints) -> Vec2 {
        constraints.constrain(Vec2::splat(8.0))
    }
    fn paint(&self, ctx: PaintContext<'_>) {
        let mut mesh = PaintMesh::new(
            [Vertex::new(self.position, Vec2::ZERO, [1.0, 0.0, 0.0, 0.0])],
            [0],
        );
        mesh.pipeline = Pipeline::Text;
        ctx.paint.add_mesh(mesh);
    }
}

#[test]
fn nested_scopes_transform_clips_preserve_alpha_and_restore_for_siblings() {
    let mut yak = Yakui::new();
    yak.set_surface_size(Vec2::splat(100.0));
    yak.set_unscaled_viewport(Rect::from_pos_size(Vec2::ZERO, Vec2::splat(100.0)));
    yak.start();
    let outer = yak.dom().begin_widget::<Scope>((
        PaintTransform {
            scale: Vec2::splat(2.0),
            translation: Vec2::new(10.0, 20.0),
            ..Default::default()
        },
        0.5,
    ));
    let inner = yak.dom().begin_widget::<Scope>((
        PaintTransform {
            scale: Vec2::splat(0.5),
            translation: Vec2::new(2.0, 3.0),
            origin: Vec2::splat(4.0),
        },
        0.5,
    ));
    let clip = yak.dom().begin_widget::<Clip>(());
    let point = yak.dom().do_widget::<Point>(Vec2::new(2.25, 3.25));
    yak.dom().end_widget::<Clip>(clip.id);
    yak.dom().end_widget::<Scope>(inner.id);
    yak.dom().end_widget::<Scope>(outer.id);
    yak.dom().do_widget::<Point>(Vec2::new(4.3, 6.1));
    yak.finish();
    assert_eq!(
        yak.layout_dom().get(point.id).unwrap().rect.size(),
        Vec2::splat(8.0)
    );
    let calls = &yak.paint().layers()[0].calls;
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0].clip,
        Some(Rect::from_pos_size(Vec2::new(18.0, 30.0), Vec2::splat(8.0)))
    );
    assert_eq!(
        calls[0].vertices[0].position,
        Vec2::new(20.25, 33.25) / 100.0
    );
    assert_eq!(calls[0].vertices[0].opacity, 0.25);
    assert_eq!(calls[0].vertices[0].color.w, 0.0);
    assert_eq!(calls[1].vertices[0].position, Vec2::new(4.0, 6.0) / 100.0);
    assert_eq!(calls[1].vertices[0].opacity, 1.0);
    assert_eq!(calls[1].clip, None);
}

#[test]
fn frame_timestamp_is_shared_by_widgets_and_advances_only_at_start() {
    let mut yak = Yakui::new();
    yak.start();
    let timestamp = yak.dom().frame_time();
    yak.dom().do_widget::<Point>(Vec2::ZERO);
    assert_eq!(yak.dom().frame_time(), timestamp);
    yak.finish();
    assert_eq!(yak.dom().frame_time(), timestamp);
    yak.start();
    assert!(yak.dom().frame_time() >= timestamp);
    yak.finish();
}
