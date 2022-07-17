mod paint_dom;
mod texture;

use glam::{Vec2, Vec4};

use crate::geometry::{Color3, Rect};
use crate::id::TextureId;

pub use self::paint_dom::*;
pub use self::texture::*;

#[non_exhaustive]
pub struct PaintRect {
    pub rect: Rect,
    pub color: Color3,
    pub texture: Option<(TextureId, Rect)>,
    pub pipeline: Pipeline,
}

impl PaintRect {
    pub fn new(rect: Rect) -> Self {
        Self {
            rect,
            color: Color3::WHITE,
            texture: None,
            pipeline: Pipeline::Main,
        }
    }
}

#[non_exhaustive]
pub struct PaintMesh<V, I> {
    pub vertices: V,
    pub indices: I,
    pub texture: Option<(TextureId, Rect)>,
    pub pipeline: Pipeline,
}

impl<V, I> PaintMesh<V, I>
where
    V: IntoIterator<Item = Vertex>,
    I: IntoIterator<Item = u16>,
{
    pub fn new(vertices: V, indices: I) -> Self {
        Self {
            vertices,
            indices,
            texture: None,
            pipeline: Pipeline::Main,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub struct PaintCall {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub texture: Option<TextureId>,
    pub pipeline: Pipeline,
}

impl PaintCall {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            texture: None,
            pipeline: Pipeline::Main,
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub struct Vertex {
    pub position: Vec2,
    pub texcoord: Vec2,
    pub color: Vec4,
}

impl Vertex {
    pub fn new<P, T, C>(position: P, texcoord: T, color: C) -> Self
    where
        P: Into<Vec2>,
        T: Into<Vec2>,
        C: Into<Vec4>,
    {
        Self {
            position: position.into(),
            texcoord: texcoord.into(),
            color: color.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pipeline {
    Main,
    Text,
}
