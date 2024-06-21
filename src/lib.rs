#![feature(unboxed_closures, fn_traits)]
use std::fmt::{Display, Formatter};
use dyn_clone::DynClone;

const Δ:f64 = 5e-6;

// ----- VECTORS -----
#[derive(Copy, Clone)]
pub struct Vector2 {
    pub x:f64,
    pub y:f64
}
#[derive(Copy, Clone)]
pub struct Vector3 {
    pub x:f64,
    pub y:f64,
    pub z:f64
}
impl Vector2 {
    pub fn new(x:f64, y:f64) -> Self {
        Self { x, y }
    }
    pub fn modulus(&self) -> f64 {
        (self.x.powi(2)+self.y.powi(2)).sqrt()
    }
    pub fn get_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }
}
impl Vector3 {
    pub fn new(x:f64, y:f64, z:f64) -> Self {
        Self { x, y, z}
    }
    pub fn modulus(&self) -> f64 {
        (self.x.powi(2)+self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    pub fn get_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
}
// Dot products for Vector2,3 where moved into Vector
// Cross product for Vector2,3
impl std::ops::Rem for Vector2 {
    type Output = f64;
    fn rem(self, rhs: Self) -> Self::Output {
        self.x*rhs.y - self.y*rhs.x
    }
}
impl std::ops::Rem for Vector3 {
    type Output = Vector3;
    fn rem(self, rhs: Self) -> Self::Output {
        Vector3 {
            x: self.y*rhs.z - self.z*rhs.y,
            y: self.z*rhs.x - self.x*rhs.z,
            z: self.x*rhs.y - self.y*rhs.x,
        }
    }
}
impl Display for Vector2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}⟩", self.x, self.y)
    }
}
impl Display for Vector3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}, {:.5}⟩", self.x, self.y, self.z)
    }
}
#[derive(Copy, Clone)]
pub enum Vector {
    TwoD(Vector2),
    ThreeD(Vector3)
}
impl Vector {
    pub fn new_2d(x:f64, y:f64) -> Self {
        Vector::TwoD(Vector2 { x, y, })
    }
    pub fn new_3d(x:f64, y:f64, z:f64) -> Self {
        Vector::ThreeD(Vector3 { x, y, z })
    }
    pub fn x(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.x,
            Vector::ThreeD(v) => v.x
        }
    }
    pub fn y(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.y,
            Vector::ThreeD(v) => v.y
        }
    }
    fn z(&self) -> f64 {
        match self {
            Vector::TwoD(_) => 0.,
            Vector::ThreeD(v) => v.z
        }
    }
}
// Dot product for Vector
impl std::ops::Mul for Vector {
    type Output = f64;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Vector::TwoD(v1), Vector::TwoD(v2)) => v1.x*v2.x+v1.y*v2.y,
            (Vector::ThreeD(v1), Vector::ThreeD(v2)) => v1.x*v2.x+v1.y*v2.y+v1.z*v2.z,
            _ => panic!("No dot product for dimensions 2x3"),
        }
    }
}
// Scalar Product
impl std::ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        match self {
            Vector::TwoD(v) => Vector::TwoD(Vector2 {
                x: v.x * scalar,
                y: v.y * scalar,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(Vector3 {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar,
            }),
        }
    }
}
impl std::ops::Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, v: Vector) -> Self::Output {
        match v {
            Vector::TwoD(v) => Vector::TwoD(Vector2 {
                x: v.x * self,
                y: v.y * self,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(Vector3 {
                x: v.x * self,
                y: v.y * self,
                z: v.z * self,
            }),
        }
    }
}
impl std::ops::Mul<i32> for Vector {
    type Output = Self;

    fn mul(self, scalar: i32) -> Self::Output {
        let scalar = scalar as f64;
        match self {
            Vector::TwoD(v) => Vector::TwoD(Vector2 {
                x: v.x * scalar,
                y: v.y * scalar,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(Vector3 {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar,
            }),
        }
    }
}
impl std::ops::Mul<Vector> for i32 {
    type Output = Vector;

    fn mul(self, v: Vector) -> Self::Output {
        match v {
            Vector::TwoD(v) => Vector::TwoD(Vector2 {
                x: v.x * self as f64,
                y: v.y * self as f64,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(Vector3 {
                x: v.x * self as f64,
                y: v.y * self as f64,
                z: v.z * self as f64,
            }),
        }
    }
}
// Cross product for Vector
impl std::ops::Rem for Vector {
    type Output = Vector;
    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Vector::TwoD(v1), Vector::TwoD(v2)) => Vector::ThreeD(Vector3::new(0., 0., v1%v2)),
            (Vector::ThreeD(v1), Vector::ThreeD(v2)) => Vector::ThreeD(v1%v2),
            (Vector::TwoD(v1), Vector::ThreeD(v2)) => Vector::ThreeD(Vector3::new(v1.x, v1.y, 0.)%v2),
            (Vector::ThreeD(v1), Vector::TwoD(v2)) => Vector::ThreeD(v1%Vector3::new(v2.x, v2.y, 0.)),
        }
    }
}
impl Display for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Vector::TwoD(v) => write!(f, "{}", v),
            Vector::ThreeD(v) => write!(f, "{}", v)
        }
    }
}
// Modulus for Vectors
pub fn modulus(v:&Vector) -> f64 {
    match v {
        Vector::TwoD(v) => v.modulus(),
        Vector::ThreeD(v) => v.modulus()
    }
}
/// The `!` operator takes the modulus of a vector
impl std::ops::Not for Vector {
    type Output = f64;

    fn not(self) -> Self::Output {
        modulus(&self)
    }
}
#[macro_export]
macro_rules! vector {
    ($x:expr, $y:expr) => {Vector::new_2d($x as f64, $y as f64)};
    ($x:expr, $y:expr, $z:expr) => {Vector::new_3d($x as f64, $y as f64, $z as f64)};
}
#[macro_export]
macro_rules! md {
    ($v:expr) => {modulus(&$v)};
}

// ----- SCALAR FUNCTIONS -----
pub trait F2DClone: DynClone + Fn(f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64) -> f64;
}
pub trait F3DClone: DynClone + Fn(f64, f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64, z:f64) -> f64;
}
impl<F> F2DClone for F
    where F: 'static + Fn(f64, f64) -> f64 + Clone,
{
    fn call(&self, x: f64, y: f64) -> f64 {
        self(x, y)
    }
}
impl<F> F3DClone for F
    where F: 'static + Fn(f64, f64, f64) -> f64 + Clone
{
    fn call(&self, x: f64, y: f64, z:f64) -> f64 {
        self(x, y, z)
    }
}
dyn_clone::clone_trait_object!(F2DClone<Output=f64>);
dyn_clone::clone_trait_object!(F3DClone<Output=f64>);
pub struct Function2D {
    pub f:Box<dyn F2DClone<Output=f64>>,
    pub expression:String
}
impl Function2D {
    fn call(&self, x:f64, y:f64) -> f64 {
        (self.f)(x, y)
    }
}
pub struct Function3D {
    pub f:Box<dyn F3DClone<Output=f64>>,
    pub expression:String
}
impl Function3D {
    fn call(&self, x:f64, y:f64, z:f64) -> f64 {
        (self.f)(x, y, z)
    }
}
impl Clone for Function2D {
    fn clone(&self) -> Function2D {
        Function2D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
    }
}
impl Clone for Function3D {
    fn clone(&self) -> Function3D {
        Function3D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
    }
}
pub enum Function {
    TwoD(Function2D),
    ThreeD(Function3D)
}
impl Function {
    pub fn expression(&self) -> String {
        match self {
            Function::TwoD(f) => f.clone().expression,
            Function::ThreeD(f) => f.clone().expression
        }
    }
}
pub fn ddx_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::TwoD(f) => {
            (f.call(args[0] + Δ, args[1])-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0] + Δ, args[1], args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
pub fn ddy_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::TwoD(f) => {
            (f.call(args[0], args[1] + Δ)-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0], args[1] + Δ, args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
pub fn ddz_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D function")
        }
        Function::ThreeD(f) => {
            (f.call(args[0], args[1], args[2]+ Δ)-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
#[macro_export]
macro_rules! ddx {
    ($f:expr, $x:expr, $y:expr) => {ddx_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddy {
    ($f:expr, $x:expr, $y:expr) => {ddy_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddz {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddz_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
impl Clone for Function {
    fn clone(&self) -> Function {
        match self {
            Function::TwoD(f) => Function::TwoD(f.clone()),
            Function::ThreeD(f) => Function::ThreeD(f.clone())
        }
    }
}
#[macro_export]
macro_rules! f {
    ($x:ident, $y:ident, $f:expr) => {
        Function::TwoD(Function2D {
            f: Box::new(|$x:f64, $y:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
    ($x:ident, $y:ident, $z:ident, $f:expr) => {
        Function::ThreeD(Function3D {
            f: Box::new(|$x:f64, $y:f64, $z:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
}
impl FnOnce<(f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl Fn<(f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl FnMut<(f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl FnOnce<(f64, f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}
impl Fn<(f64, f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}
impl FnMut<(f64, f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}

// ----- VECTOR FUNCTIONS -----
pub struct VectorFunction2D {
    pub f1:Box<dyn Fn(f64, f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64, f64) -> f64>,
    pub expression_f2:String
}
pub struct VectorFunction3D {
    pub f1:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f2:String,
    pub f3:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f3:String
}
pub enum VectorFunction {
    TwoD(VectorFunction2D),
    ThreeD(VectorFunction3D)
}
impl Display for VectorFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorFunction::TwoD(v) => {
                write!(f, "⟨{:.5}, {:.5}⟩", v.expression_f1, v.expression_f2)
            },
            VectorFunction::ThreeD(v) => {
                write!(f, "⟨{:.5}, {:.5}, {:.5}⟩", v.expression_f1, v.expression_f2, v.expression_f3)
            }
        }
    }
}
#[macro_export]
macro_rules! vector_function {
    ($x:ident, $y:ident, $f1:expr, $f2:expr) => {
        VectorFunction::TwoD(VectorFunction2D {
            f1: Box::new(|$x:f64, $y:f64| $f1),
            expression_f1: String::from(stringify!($f1)),
            f2: Box::new(|$x:f64, $y:f64| $f2),
            expression_f2: String::from(stringify!($f2))
        })
    };
    ($x:ident, $y:ident, $z:ident, $f1:expr, $f2:expr, $f3:expr) => {
        VectorFunction::ThreeD(VectorFunction3D {
            f1: Box::new(|$x:f64, $y:f64, $z:f64| $f1),
            expression_f1: String::from(stringify!($f1)),
            f2: Box::new(|$x:f64, $y:f64, $z:f64| $f2),
            expression_f2: String::from(stringify!($f2)),
            f3: Box::new(|$x:f64, $y:f64, $z:f64| $f3),
            expression_f3: String::from(stringify!($f3))
        })
    };
}
impl FnOnce<(f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl Fn<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl FnMut<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl FnOnce<(f64, f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
impl Fn<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
impl FnMut<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
// Partial Derivatives
pub fn ddx_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => {
            Vector::TwoD(Vector2::new(((v.f1)(args[0] + Δ, args[1])-(v.f1)(args[0], args[1]))/Δ,((v.f2)(args[0] + Δ, args[1])-(v.f2)(args[0], args[1]))/Δ))
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(Vector2::new(((v.f1)(args[0] + Δ, args[1], args[2])-(v.f1)(args[0], args[1], args[2]))/Δ,((v.f2)(args[0] + Δ, args[1], args[2])-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
pub fn ddy_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => {
            Vector::TwoD(Vector2::new(((v.f1)(args[0], args[1] + Δ)-(v.f1)(args[0], args[1]))/Δ,((v.f2)(args[0], args[1] + Δ)-(v.f2)(args[0], args[1]))/Δ))
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(Vector2::new(((v.f1)(args[0], args[1] + Δ, args[2])-(v.f1)(args[0], args[1], args[2]))/Δ,((v.f2)(args[0], args[1] + Δ, args[2])-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
pub fn ddz_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D vector function")
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(Vector2::new(((v.f1)(args[0], args[1], args[2] + Δ)-(v.f1)(args[0], args[1], args[2]))/Δ,((v.f2)(args[0], args[1], args[2] + Δ)-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
#[macro_export]
macro_rules! dvdx {
    ($f:expr, $x:expr, $y:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! dvdy {
    ($f:expr, $x:expr, $y:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! dvdz {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddz_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
// Curl
pub fn curl(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            let (x, y) = (args[0], args[1]);
            vector!(0, 0, ddx_v(v, vec![x, y]).y() - ddy_v(v, vec![x, y]).x())
        },
        VectorFunction::ThreeD(_) => {
            let (x, y, z) = (args[0], args[1], args[2]);
            vector!(ddy_v(v, vec![x, y, z]).z() - ddz_v(v, vec![x, y, z]).y(), ddz_v(v, vec![x, y, z]).x() - ddx_v(v, vec![x, y, z]).z(), ddx_v(v, vec![x, y, z]).y() - ddy_v(v, vec![x, y, z]).x())
        },
    }
}
#[macro_export]
macro_rules! curl {
    ($v:expr, $x:expr, $y:expr) => {
        curl(&$v, vec![$x as f64, $y as f64])
    };
    ($v:expr, $x:expr, $y:expr, $z:expr) => {
        curl(&$v, vec![$x as f64, $y as f64, $z as f64])
    };
}
// Div
pub fn div(v:&VectorFunction, args:Vec<f64>) -> f64 {
    match v {
        VectorFunction::TwoD(_) => {
            let (x, y) = (args[0], args[1]);
            ddx_v(v, vec![x, y]).x() + ddy_v(v, vec![x, y]).y()
        },
        VectorFunction::ThreeD(_) => {
            let (x, y, z) = (args[0], args[1], args[2]);
            ddx_v(v, vec![x, y, z]).x() + ddy_v(v, vec![x, y, z]).y() + ddz_v(v, vec![x, y, z]).z()
        }
    }
}
#[macro_export]
macro_rules! div {
    ($v:expr, $x:expr, $y:expr) => {
        div(&$v, vec![$x as f64, $y as f64])
    };
    ($v:expr, $x:expr, $y:expr, $z:expr) => {
        div(&$v, vec![$x as f64, $y as f64, $z as f64])
    };
}

// ----- GRADIENT -----
pub fn grad(f:&Function) -> VectorFunction {
    match f {
        Function::TwoD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            VectorFunction::TwoD(VectorFunction2D {
                expression_f1: format!("ddx({})", f.expression()),
                expression_f2: format!("ddy({})", f.expression()),
                f1: Box::new(move |x:f64, y:f64| ddx_s(&f1, vec![x, y])),
                f2: Box::new(move |x:f64, y:f64| ddy_s(&f2, vec![x, y]))
            })
        }
        Function::ThreeD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            let f3 = f.clone();
            VectorFunction::ThreeD(VectorFunction3D {
                expression_f1: format!("ddx({})", f.expression()),
                expression_f2: format!("ddy({})", f.expression()),
                expression_f3: format!("ddz({})", f.expression()),
                f1: Box::new(move |x:f64, y:f64, z:f64| ddx_s(&f1, vec![x, y, z])),
                f2: Box::new(move |x:f64, y:f64, z:f64| ddy_s(&f2, vec![x, y, z])),
                f3: Box::new(move |x:f64, y:f64, z:f64| ddz_s(&f3, vec![x, y, z])),
            })
        }
    }
}
#[macro_export]
macro_rules! grad {
    ($f:expr) => {
        grad(&$f)
    }
}

// ----- PARAMETRIC CURVES -----

// ----- SETS -----

// ----- CONTOURS -----

// ----- LIMITS -----

// ----- LINE INTEGRAL -----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors() {
        let u = vector!(3, 4);
        let v = vector!(4, 3);
        let w = u%v;
        println!("2*(u%v) = {}, {}", md!(2*w), u.z());
        assert_eq!(2.*u*v, 48.);
    }

    //noinspection ALL //This is for the squiggly lines when evaluating Functions
    #[test]
    fn scalar_functions() {
        let f:Function = f!(x, y, 2.*x*y);
        let g = f!(x, y, z, x.powi(2)*y + z);
        assert_eq!(f(2., 2.), 8.);
        assert_eq!(g(1., 2., 2.), 4.);
        println!("df/dx(2,2) = {:.6}", ddx!(f, 0, 2));
        assert_eq!(g.expression(), String::from("x.powi(2) * y + z"));
    }

    //noinspection ALL //This is for the squiggly lines when evaluating Vector Functions
    #[test]
    fn vector_functions() {
        let F:VectorFunction = vector_function!(x, y, -y, x);
        println!("F(1,2) = {:.5}", F(1., 2.));
        println!("∂F/∂y = {}", dvdy!(F, 1, 2));
        println!("|∇xF(1, 2)| = {:.5}", !curl!(F, 1, 2));

        let g = f!(x, y, z, x + y +z);
        let del_g = grad!(g);
        println!("∇g(1, 2, 3) = {}", del_g(1., 2., 3.))
    }
}
