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
pub fn ddx(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::TwoD(f) => {
            (f.call(args[0] + Δ, args[1])-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0] + Δ, args[1], args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
pub fn ddy(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::TwoD(f) => {
            (f.call(args[0], args[1] + Δ)-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0], args[1] + Δ, args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
pub fn ddz(f:&Function, args:Vec<f64>) -> f64 {
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
    ($f:expr, $x:expr, $y:expr) => {ddx(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddy {
    ($f:expr, $x:expr, $y:expr) => {ddy(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddz {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddz(&$f, vec![$x as f64, $y as f64, $z as f64])};
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

    //noinspection ALL //This is for the squiggly lines when evaluating f and g
    #[test]
    fn scalar_functions() {
        let f:Function = f!(x, y, 2.*x*y);
        let g = f!(x, y, z, x.powi(2)*y + z);
        assert_eq!(f(2., 2.), 8.);
        assert_eq!(g(1., 2., 2.), 4.);
        println!("df/dx(2,2) = {:.6}", ddx!(f, 0, 2));
        assert_eq!(g.expression(), String::from("x.powi(2) * y + z"));
    }

    #[test]
    fn vector_functions() {

    }
}
