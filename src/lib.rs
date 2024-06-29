#![feature(unboxed_closures, fn_traits)]
//! This crate is aimed to be an easy-to-use library for all calculations related to multivariable
//! and vector calculus in Rust, and is able to compute a wide variety of operations numerically, doing so
//! relatively fast. \
//! To achieve this, it contains a growing number of macros designed to create different objects, functions
//! and perform operations. For learning about all of these, you can go to the docs.rs page for this crate. \
//! ## Learn
//! The recommended order to learn the library is
//! 1. [Vector] - Vectors
//! 2. [Function] - Scalar functions
//! 3. [VectorFunction] - Vector functions
//! 4. [ParametricCurve], [Set], [Contour] - Contours
//! 5. [ParametricSurface], [FSet], [Surface] - Surfaces
//! 6. [integral!], [line_integral!], [surface_integral!] - Integrate everything!!
//!
//! As extras, also take into account [grad!], [curl!], [div!], derivatives like [ddx!] and [ddyv!], and lastly but
//! not least important: the [setup!] macro.
//! ## About the crate
//! This crate was made by a 4th semester physics student to help verify the analytic results obtained in excercises,
//! but is also intended for use in other purposes as well. If you have any issues or ideas, be sure to leave them
//! in this crate's GitHub page: [https://github.com/OkiMusix05/rust_vector_calculus].
use std::fmt::{Display, Formatter};
use dyn_clone::DynClone;
//use std::cmp::{min, max};
use rand::Rng;
// ----- IDEAS -----
// - Change of variables (Jacobian)
// - Solve equations for Function = 0
// - Lagrange multipliers
// - Flux Lines for Vector Functions
// - Differential equations solver

// ----- GLOBAL CONSTANTS -----
/// # General Error
/// The error Δ = 5*10^-6
pub const Δ:f64 = 5e-6;
/// # Universal gravitational constant
/// The constant G has a value of 6.6743*10^-11
/// \[G\] = m^3/kg*s^2
const G:f64 = 6.6743e-11;
/// # Speed of light in a vacuum
/// The constant c has a value of 299,792,458
/// \[c\] = m/s
const C:f64 = 299_792_458.0;

// ----- VECTORS -----
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct _Vector2 {
    pub x:f64,
    pub y:f64
}
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct _Vector3 {
    pub x:f64,
    pub y:f64,
    pub z:f64
}
impl _Vector2 {
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
impl _Vector3 {
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
impl std::ops::Rem for _Vector2 {
    type Output = f64;
    fn rem(self, rhs: Self) -> Self::Output {
        self.x*rhs.y - self.y*rhs.x
    }
}
impl std::ops::Rem for _Vector3 {
    type Output = _Vector3;
    fn rem(self, rhs: Self) -> Self::Output {
        _Vector3 {
            x: self.y*rhs.z - self.z*rhs.y,
            y: self.z*rhs.x - self.x*rhs.z,
            z: self.x*rhs.y - self.y*rhs.x,
        }
    }
}
impl Display for _Vector2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}⟩", self.x, self.y)
    }
}
impl Display for _Vector3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{:.5}, {:.5}, {:.5}⟩", self.x, self.y, self.z)
    }
}
#[derive(Copy, Clone, Debug)]
/// A mathematical vector in R2 or R3.
/// # Vector
/// Vectors can be created using the [`vector!`] macro, which depends on the number of
/// arguments passed for the dimension of the vector, although they all fall into `Vector`.
/// ### Methods
/// - Vectors can be multiplied using the `*` operator, which performs the dot product,
/// or the `%` operator, which always returns a 3D vector with its cross product. \
/// These operations do not take ownership of the vectors, although if you want to you can clone
/// them using `v.clone()`.
/// - For taking the modulus (length) of a vector you can use the [md!] macro or the `!` operator,
/// which in front of a vector takes its length.
/// - You can access a coordinate from a vector with the `x`, `y` and `z` methods, like `u.x()`.
/// - Vectors implement display and look like this: ⟨3.00000, 4.00000⟩, using \langle and \rangle
/// from unicode for around them
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let u:Vector = vector!(3, 4);
/// let v:Vector = vector!(0, 1);
///
/// assert_eq!(u%v, vector!(0, 0, 3)); // Cross product
/// assert_eq!(u*v, 4.); // Dot product
/// assert_eq!(2.*u, vector!(6, 8)); // Product by scalar
/// assert_eq!(!u, 5.); // Length of the vector
/// ```
pub enum Vector {
    TwoD(_Vector2),
    ThreeD(_Vector3)
}
impl Vector {
    /// Creates a 2d vector
    pub fn new_2d(x:f64, y:f64) -> Self {
        Vector::TwoD(_Vector2 { x, y, })
    }
    /// Creates a 3d vector
    pub fn new_3d(x:f64, y:f64, z:f64) -> Self {
        Vector::ThreeD(_Vector3 { x, y, z })
    }
    /// Returns the x coordinate
    pub fn x(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.x,
            Vector::ThreeD(v) => v.x
        }
    }
    /// Returns the y coordinate
    pub fn y(&self) -> f64 {
        match self {
            Vector::TwoD(v) => v.y,
            Vector::ThreeD(v) => v.y
        }
    }
    /// Returns the z coordinate and 0 for 2D vectors
    pub fn z(&self) -> f64 {
        match self {
            Vector::TwoD(_) => 0.,
            Vector::ThreeD(v) => v.z
        }
    }
}
impl PartialEq for Vector {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Vector::TwoD(u), Vector::TwoD(v)) => if u.x == v.x && u.y == v.y { true } else {false},
            (Vector::ThreeD(u), Vector::ThreeD(v)) => if u.x == v.x && u.y == v.y && u.z == v.z { true } else { false },
            (_, _) => false
        }
    }
}
/// Dot product for Vector
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
/// Multiplication by scalar of a vector
impl std::ops::Mul<f64> for Vector {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        match self {
            Vector::TwoD(v) => Vector::TwoD(_Vector2 {
                x: v.x * scalar,
                y: v.y * scalar,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(_Vector3 {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar,
            }),
        }
    }
}
/// Multiplication by scalar of a vector
impl std::ops::Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, v: Vector) -> Self::Output {
        match v {
            Vector::TwoD(v) => Vector::TwoD(_Vector2 {
                x: v.x * self,
                y: v.y * self,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(_Vector3 {
                x: v.x * self,
                y: v.y * self,
                z: v.z * self,
            }),
        }
    }
}
/// Multiplication by scalar of a vector
impl std::ops::Mul<i32> for Vector {
    type Output = Self;

    fn mul(self, scalar: i32) -> Self::Output {
        let scalar = scalar as f64;
        match self {
            Vector::TwoD(v) => Vector::TwoD(_Vector2 {
                x: v.x * scalar,
                y: v.y * scalar,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(_Vector3 {
                x: v.x * scalar,
                y: v.y * scalar,
                z: v.z * scalar,
            }),
        }
    }
}
/// Multiplication by scalar of a vector
impl std::ops::Mul<Vector> for i32 {
    type Output = Vector;

    fn mul(self, v: Vector) -> Self::Output {
        match v {
            Vector::TwoD(v) => Vector::TwoD(_Vector2 {
                x: v.x * self as f64,
                y: v.y * self as f64,
            }),
            Vector::ThreeD(v) => Vector::ThreeD(_Vector3 {
                x: v.x * self as f64,
                y: v.y * self as f64,
                z: v.z * self as f64,
            }),
        }
    }
}
/// Cross product for Vector
impl std::ops::Rem for Vector {
    type Output = Vector;
    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Vector::TwoD(v1), Vector::TwoD(v2)) => Vector::ThreeD(_Vector3::new(0., 0., v1%v2)),
            (Vector::ThreeD(v1), Vector::ThreeD(v2)) => Vector::ThreeD(v1%v2),
            (Vector::TwoD(v1), Vector::ThreeD(v2)) => Vector::ThreeD(_Vector3::new(v1.x, v1.y, 0.)%v2),
            (Vector::ThreeD(v1), Vector::TwoD(v2)) => Vector::ThreeD(v1% _Vector3::new(v2.x, v2.y, 0.)),
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
#[doc(hidden)]
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
/// Creates a vector
/// # Vector macro
/// The vector macro can make a 2D or 3D [Vector] depending on the number of arguments passed.\
/// To create a macro it can take i32's or f64's but that's just for convenience as they all
/// turn into f64's inside the vector.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// use std::f64::consts::PI;
/// let u:Vector = vector!(0, 1);
/// let v:Vector = vector!(1, PI, PI/2.);
/// ```
#[macro_export]
macro_rules! vector {
    ($x:expr, $y:expr) => {Vector::new_2d($x as f64, $y as f64)};
    ($x:expr, $y:expr, $z:expr) => {Vector::new_3d($x as f64, $y as f64, $z as f64)};
}
/// Takes the length of a vector
/// # Modulus macro
/// This macro takes the length of any [Vector], although it is often easier using the `!` operator.
/// ```
/// use vector_calculus::*;
/// let v:Vector = vector!(0, 3, 4);
/// assert_eq!(md!(v), 5.);
/// ```
#[macro_export]
macro_rules! md {
    ($v:expr) => {modulus(&$v)};
}

// ----- SCALAR FUNCTIONS -----
#[doc(hidden)]
pub trait F1DClone: DynClone + Fn(f64,) -> f64 {
    fn call(&self, x: f64) -> f64;
}
#[doc(hidden)]
pub trait F2DClone: DynClone + Fn(f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64) -> f64;
}
#[doc(hidden)]
pub trait F3DClone: DynClone + Fn(f64, f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64, z:f64) -> f64;
}
impl<F> F1DClone for F
    where F: 'static + Fn(f64,) -> f64 + Clone
{
    fn call(&self, x: f64) -> f64 {
        self(x)
    }
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
dyn_clone::clone_trait_object!(F1DClone<Output=f64>);
dyn_clone::clone_trait_object!(F2DClone<Output=f64>);
dyn_clone::clone_trait_object!(F3DClone<Output=f64>);
#[doc(hidden)]
pub struct _Function1D {
    pub f:Box<dyn F1DClone<Output=f64>>,
    pub expression:String
}
impl _Function1D {
    fn call(&self, x:f64) -> f64 {
        (self.f)(x)
    }
}
#[doc(hidden)]
pub struct _Function2D {
    pub f:Box<dyn F2DClone<Output=f64>>,
    pub expression:String
}
impl _Function2D {
    fn call(&self, x:f64, y:f64) -> f64 {
        (self.f)(x, y)
    }
}
#[doc(hidden)]
pub struct _Function3D {
    pub f:Box<dyn F3DClone<Output=f64>>,
    pub expression:String
}
impl _Function3D {
    fn call(&self, x:f64, y:f64, z:f64) -> f64 {
        (self.f)(x, y, z)
    }
}
impl Clone for _Function1D {
    fn clone(&self) -> _Function1D {
        _Function1D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
    }
}
impl Clone for _Function2D {
    fn clone(&self) -> _Function2D {
        _Function2D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
    }
}
impl Clone for _Function3D {
    fn clone(&self) -> _Function3D {
        _Function3D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
    }
}
/// Functions of one, two and three variables
/// # Function
/// Functions can be created using the [`f!`] macro, and you can evaluate them like any rust function with
/// the corresponding number of arguments, all of them f64's. \
/// As a bonus, you can also evaluate an n-sized [Vector] in an n-variable function, and it will work as intended. \
/// They also save their expression as a string, which you can obtain as `f.expression()`.\
/// You can use the [ddx!], [ddy!] and [ddz!] macros to take its derivative at a specific point, and use
/// the [integral!] macro to evaluate its integral.\
/// For taking its limit you can use the [limit!] macro, which returns a number. \
/// Finally, you can make a [VectorFunction] out of a two or three-dimensional function using
/// the [grad!] macro, which returns the gradient vector function. \
/// Functions can also be cloned.
/// _Note_: You may get a syntax error for using a Function as a function, but it will compile correctly.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, y, x*y); // Two variable function
/// let g:Function = f!(x, x.powi(2)); // Single variable function
/// assert_eq!(f(1., 2.), 2.);
/// assert_eq!(g(4.), 16.0);
/// ```
pub enum Function {
    OneD(_Function1D),
    TwoD(_Function2D),
    ThreeD(_Function3D)
}
impl Function {
    /// Returns the expression of a Function as a string
    pub fn expression(&self) -> String {
        match self {
            Function::OneD(f) => f.clone().expression,
            Function::TwoD(f) => f.clone().expression,
            Function::ThreeD(f) => f.clone().expression
        }
    }
}
#[doc(hidden)]
pub fn ddx_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::OneD(f) => {
            (f.call(args[0] + Δ)-f.call(args[0]))/Δ
        }
        Function::TwoD(f) => {
            (f.call(args[0] + Δ, args[1])-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0] + Δ, args[1], args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
#[doc(hidden)]
pub fn ddy_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::OneD(f) => {
            panic!("Can't take partial with respect to y of a 1D function")
        }
        Function::TwoD(f) => {
            (f.call(args[0], args[1] + Δ)-f.call(args[0], args[1]))/Δ
        }
        Function::ThreeD(f) => {
            (f.call(args[0], args[1] + Δ, args[2])-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
#[doc(hidden)]
pub fn ddz_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::OneD(_) => {
            panic!("Can't take partial with respect to z of a 1D function")
        }
        Function::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D function")
        }
        Function::ThreeD(f) => {
            (f.call(args[0], args[1], args[2]+ Δ)-f.call(args[0], args[1], args[2]))/Δ
        }
    }
}
/// Total derivative for single-variable functions and partial x derivative for two and three-variable ones.
/// # Partial x
/// This macro takes a [Function] and n f64's where n is the number of variables in the function,
/// and returns the derivative with respect to the first variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, y, z, x*y.powi(2) + z);
/// let a:f64 = ddx!(f, 1., 2., 9.);
/// assert_eq!(a, 3.9999999998485687); // Analytically, it's 4
/// ```
#[macro_export]
macro_rules! ddx {
    ($f:expr, $x:expr) => {ddx_s(&$f, vec![$x as f64])};
    ($f:expr, $x:expr, $y:expr) => {ddx_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
/// Partial y derivative for two and three-variable functions.
/// # Partial y
/// This macro takes a [Function] and n f64's where n is the number of variables in the function,
/// and returns the derivative with respect to the second variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, y, z, x*y.powi(2) + z);
/// let b:f64 = ddy!(f, 1., 2., 9.);
/// assert_eq!(b, 4.000004999937801); // Analytically, it's 4
/// ```
#[macro_export]
macro_rules! ddy {
    ($f:expr, $x:expr, $y:expr) => {ddy_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
/// Partial z derivative for three-variable functions.
/// # Partial z
/// This macro takes a [Function] and 3 f64's representing a point in the form (x, y, z)
/// and returns the derivative with respect to the third variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, y, z, x*y.powi(2) + z);
/// let b:f64 = ddz!(f, 1., 2., 9.);
/// assert_eq!(b, 0.9999999999621422); // Analytically, it's 1
/// ```
#[macro_export]
macro_rules! ddz {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddz_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
impl Clone for Function {
    fn clone(&self) -> Function {
        match self {
            Function::OneD(f) => Function::OneD(f.clone()),
            Function::TwoD(f) => Function::TwoD(f.clone()),
            Function::ThreeD(f) => Function::ThreeD(f.clone())
        }
    }
}
/// Creates a one, two, or three-variable function
/// # Function Macro
/// To create a function you need n identifiers, which mean the variables that the function takes in,
/// and an expression involving some, all or none of the variables that the function should return.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, y, x.powi(2) + 0.5*y);
/// let g:Function = f!(u, 9.8*u);
/// ```
#[macro_export]
macro_rules! f {
    ($x:ident, $f:expr) => {
        Function::OneD(_Function1D {
            f: Box::new(|$x:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
    ($x:ident, $y:ident, $f:expr) => {
        Function::TwoD(_Function2D {
            f: Box::new(|$x:f64, $y:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
    ($x:ident, $y:ident, $z:ident, $f:expr) => {
        Function::ThreeD(_Function3D {
            f: Box::new(|$x:f64, $y:f64, $z:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
}
impl FnOnce<(f64,)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        match self {
            Function::OneD(f) => f.call(args.0),
            Function::TwoD(_) => panic!("1D function can't take 2 arguments"),
            Function::ThreeD(_) => panic!("1D function can't take 3 arguments")
        }
    }
}
impl Fn<(f64,)> for Function {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        match self {
            Function::OneD(f) => f.call(args.0),
            Function::TwoD(_) => panic!("1D function can't take 2 arguments"),
            Function::ThreeD(_) => panic!("1D function can't take 3 arguments")
        }
    }
}
impl FnMut<(f64,)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        match self {
            Function::OneD(f) => f.call(args.0),
            Function::TwoD(_) => panic!("1D function can't take 2 arguments"),
            Function::ThreeD(_) => panic!("1D function can't take 3 arguments")
        }
    }
}
impl FnOnce<(f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl Fn<(f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl FnMut<(f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(f) => f.call(args.0, args.1),
            Function::ThreeD(_) => panic!("3D function can't take 2 arguments")
        }
    }
}
impl FnOnce<(f64, f64, f64)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}
impl Fn<(f64, f64, f64)> for Function {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}
impl FnMut<(f64, f64, f64)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take 2 arguments"),
            Function::TwoD(_) => panic!("2D function can't take 3 arguments"),
            Function::ThreeD(f) => f.call(args.0, args.1, args.2)
        }
    }
}
impl FnOnce<(Vector,)> for Function {
    type Output = f64;

    extern "rust-call" fn call_once(self, args: (Vector,)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take a vector as an argument"),
            Function::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => f.call(v.x, v.y),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D function")
                }
            },
            Function::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => f.call(v.x, v.y, 0.),// panic!("2D vector passed to 3D function"),
                    Vector::ThreeD(v) => f.call(v.x, v.y, v.z)
                }
            }
        }
    }
}
impl Fn<(Vector,)> for Function {
    extern "rust-call" fn call(&self, args: (Vector,)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take a vector as an argument"),
            Function::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => f.call(v.x, v.y),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D function")
                }
            },
            Function::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(_) => panic!("2D vector passed to 3D function"),
                    Vector::ThreeD(v) => f.call(v.x, v.y, v.z)
                }
            }
        }
    }
}
impl FnMut<(Vector,)> for Function {
    extern "rust-call" fn call_mut(&mut self, args: (Vector,)) -> Self::Output {
        match self {
            Function::OneD(_) => panic!("1D function can't take a vector as an argument"),
            Function::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => f.call(v.x, v.y),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D function")
                }
            },
            Function::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(_) => panic!("2D vector passed to 3D function"),
                    Vector::ThreeD(v) => f.call(v.x, v.y, v.z)
                }
            }
        }
    }
}

// ----- LIMITS -----
#[doc(hidden)]
pub fn limit_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::OneD(_) => {
            let up = f(args[0] + Δ);
            let down = f(args[0] - Δ);
            (up + down)/2.
        },
        Function::TwoD(_) => {
            let up = f(args[0] + Δ, args[1] + Δ);
            let down = f(args[0] - Δ, args[1] - Δ);
            (up + down)/2.
        },
        Function::ThreeD(_) => {
            let up = f(args[0] + Δ, args[1] + Δ, args[2] + Δ);
            let down = f(args[0] - Δ, args[1] - Δ, args[2] - Δ);
            (up + down)/2.
        }
    }
}
/// Calculus limits for functions
/// # Limit
/// The limit macro takes in a [Function] of n variables and n f64's, representing the point to take
/// the limit at. It also uses the `=>` syntax because it fits here and looks cool.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Function = f!(x, (x.powi(2)-25.)/(x-5.)); // (x^2-25)/(x-5)
/// let a:f64 = limit!(f => 5);
/// assert_eq!(a, 10.0); // Analytically it's 10
/// ```
#[macro_export]
macro_rules! limit {
    ($f:expr => $x:expr) => {limit_s(&$f, vec![$x as f64])};
    ($f:expr => $x:expr,$y:expr) => {limit_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr => $x:expr,$y:expr, $z:expr) => {limit_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}

// ----- VECTOR FUNCTIONS -----
#[derive(Clone)]
#[doc(hidden)]
pub struct _VectorFunction2D {
    pub f1:Function,
    pub f2:Function,
    pub potential: Option<Function>,
}
#[derive(Clone)]
#[doc(hidden)]
pub struct _VectorFunction3D {
    pub f1:Function,
    pub f2:Function,
    pub f3:Function,
    pub potential: Option<Function>,
}
/// Vector functions for 2D and 3D spaces
/// # Vector Function
/// Vector functions are functions of two or three variables that return two or three-dimensional [Vector]s. \
/// You can create one using the [vector_function!] macro. \
/// Vector functions evaluate just like [Function]s and rust functions, and you can even evaluate them on [Vector]s. \
/// You can take its partial derivatives with the [ddxv!], [ddyv!] and [ddzv!] macros, adding the 'v' at the end
/// to signal that it is the partial of a vector function, and that it returns a [Vector] as such. \
/// Furthermore, you can use the [curl!] and [div!] macro to evaluate its rotational and divergence at a point. \
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let v:VectorFunction = vector_function!(x, y, -y, x);
/// assert_eq!(v(0., 1.), vector!(-1, 0)); // v(0, 1) = (-1, 0)
/// ```
#[derive(Clone)]
pub enum VectorFunction {
    TwoD(_VectorFunction2D),
    ThreeD(_VectorFunction3D)
}
impl VectorFunction {
    fn potential(&self, args:Vec<f64>) -> f64 {
        match self {
            VectorFunction::TwoD(v) => if let Some(f) = &v.potential {
                return f(args[0], args[1])
            } else {f64::NAN},
            VectorFunction::ThreeD(v) => if let Some(f) = &v.potential {
                return f(args[0],args[1], args[2])
            } else {f64::NAN}
        }
    }
    pub fn potential_expression(&self) -> String {
        match self {
            VectorFunction::TwoD(v) => if let Some(p) = v.potential.clone() {
                p.expression()
            } else {String::from("")},
            VectorFunction::ThreeD(v) => if let Some(p) = v.potential.clone() {
                p.expression()
            } else {String::from("")}
        }
    }
}
impl Display for VectorFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorFunction::TwoD(v) => {
                write!(f, "⟨{:.5}, {:.5}⟩", v.f1.expression(), v.f2.expression())
            },
            VectorFunction::ThreeD(v) => {
                write!(f, "⟨{:.5}, {:.5}, {:.5}⟩", v.f1.expression(), v.f2.expression(), v.f3.expression())
            }
        }
    }
}
/// Creates vector functions
/// # Vector Function macro
/// This macro takes in two or three identifiers (the variable names) and two or three expressions that may use
/// those variables, and outputs a [VectorFunction].
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let F:VectorFunction = vector_function!(x, y, z, y.exp(), x + z, x*y*z);
/// ```
#[macro_export]
macro_rules! vector_function {
    ($x:ident, $y:ident, $f1:expr, $f2:expr) => {
        VectorFunction::TwoD(_VectorFunction2D {
            f1: Function::TwoD(_Function2D {
                f: Box::new(|$x:f64, $y:f64| $f1),
                expression: String::from(stringify!($f1))
            }),
            f2: Function::TwoD(_Function2D {
                f: Box::new(|$x:f64, $y:f64| $f2),
                expression: String::from(stringify!($f2))
            }),
            potential: Option::None,
        })
    };
    ($x:ident, $y:ident, $z:ident, $f1:expr, $f2:expr, $f3:expr) => {
        VectorFunction::ThreeD(_VectorFunction3D {
            f1: Function::ThreeD(_Function3D {
                f: Box::new(|$x:f64, $y:f64, $z:f64| $f1),
                expression: String::from(stringify!($f1))
            }),
            f2: Function::ThreeD(_Function3D {
                f: Box::new(|$x:f64, $y:f64, $z:f64| $f2),
                expression: String::from(stringify!($f2))
            }),
            f3: Function::ThreeD(_Function3D {
                f: Box::new(|$x:f64, $y:f64, $z:f64| $f3),
                expression: String::from(stringify!($f3))
            }),
            potential: Option::None,
        })
    };
}
impl FnOnce<(f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl Fn<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl FnMut<(f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(v) => Vector::TwoD(_Vector2::new((v.f1)(args.0, args.1), (v.f2)(args.0, args.1))),
            VectorFunction::ThreeD(_) => panic!("3D vector function can't take 2 arguments")
        }
    }
}
impl FnOnce<(f64, f64, f64)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
impl Fn<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
impl FnMut<(f64, f64, f64)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64, f64)) -> Self::Output {
        match self {
            VectorFunction::TwoD(_) => panic!("2D vector function can't take 3 arguments"),
            VectorFunction::ThreeD(v) => Vector::ThreeD(_Vector3::new((v.f1)(args.0, args.1, args.2), (v.f2)(args.0, args.1, args.2), (v.f3)(args.0, args.1, args.2)))
        }
    }
}
impl FnOnce<(Vector,)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
                }
            }
        }
    }
}
impl Fn<(Vector,)> for VectorFunction {
    extern "rust-call" fn call(&self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
                }
            }
        }
    }
}
impl FnMut<(Vector,)> for VectorFunction {
    extern "rust-call" fn call_mut(&mut self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::TwoD(_Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(_Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
                }
            }
        }
    }
}
// Partial Derivatives
#[doc(hidden)]
pub fn ddx_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => {
            Vector::TwoD(_Vector2::new(((v.f1)(args[0] + Δ, args[1])-(v.f1)(args[0], args[1]))/Δ, ((v.f2)(args[0] + Δ, args[1])-(v.f2)(args[0], args[1]))/Δ))
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(_Vector2::new(((v.f1)(args[0] + Δ, args[1], args[2])-(v.f1)(args[0], args[1], args[2]))/Δ, ((v.f2)(args[0] + Δ, args[1], args[2])-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
#[doc(hidden)]
pub fn ddy_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(v) => {
            Vector::TwoD(_Vector2::new(((v.f1)(args[0], args[1] + Δ)-(v.f1)(args[0], args[1]))/Δ, ((v.f2)(args[0], args[1] + Δ)-(v.f2)(args[0], args[1]))/Δ))
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(_Vector2::new(((v.f1)(args[0], args[1] + Δ, args[2])-(v.f1)(args[0], args[1], args[2]))/Δ, ((v.f2)(args[0], args[1] + Δ, args[2])-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
#[doc(hidden)]
pub fn ddz_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D vector function")
        },
        VectorFunction::ThreeD(v) => {
            Vector::ThreeD(_Vector3::new(((v.f1)(args[0], args[1], args[2] + Δ)-(v.f1)(args[0], args[1], args[2]))/Δ, ((v.f2)(args[0], args[1], args[2] + Δ)-(v.f2)(args[0], args[1], args[2]))/Δ, ((v.f3)(args[0], args[1], args[2] + Δ)-(v.f3)(args[0], args[1], args[2]))/Δ))
        }
    }
}
/// Partial x for a vector function
/// # Vector Partial x
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the first variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let F:VectorFunction = vector_function!(x, y, -2.*y, x);
/// let v:Vector = ddxv!(F, 2, 1.);
/// assert_eq!(v, vector!(0, 0.9999999999621422)); // Analytically, it's (0, 1)
/// ```
#[macro_export]
macro_rules! ddxv {
    ($f:expr, $x:expr, $y:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
/// Partial y for a vector function
/// # Vector Partial y
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the second variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let F:VectorFunction = vector_function!(x, y, -2.*y, x);
/// let v:Vector = ddyv!(F, 2, 1.);
/// assert_eq!(v, vector!(-2.0000000000131024, 0)); // Analytically, it's (-2, 0)
/// ```
#[macro_export]
macro_rules! ddyv {
    ($f:expr, $x:expr, $y:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
/// Partial z for a vector function
/// # Vector Partial z
/// This macro takes a [VectorFunction] and n f64's representing a point
/// and returns the derivative with respect to the third variable at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let F:VectorFunction = vector_function!(x, y, z, x, y, z.powi(2));
/// let v:Vector = ddzv!(F, 2, 1, 2);
/// assert_eq!(v, vector!(0, 0, 4.000004999760165)); // Analytically, it's (0, 0, 4)
/// ```
#[macro_export]
macro_rules! ddzv {
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddz_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
// Curl
#[doc(hidden)]
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
/// Curl of a vector function at a point
/// # Curl
/// The curl macro takes in a [VectorFunction] and n f64's representing the point to evaluate, and
/// returns a vector with the result of applying the rotational operator to it.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f = vector_function!(x, y, -y, x);
/// let c:Vector = curl!(f, 2, 3);
/// assert_eq!(c, vector!(0, 0, 1.9999999999242843)) // Analytically its (0, 0, 2)
/// ```
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
#[doc(hidden)]
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
/// Divergence of a vector function at a point
/// # Divergence
/// The divergence macro takes a [VectorFunction] and n f64's representing the point to evaluate,
/// and returns a f64 as the result of applying the divergence operator to the function at that point.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f = vector_function!(x, y, 2.*x, 2.*y);
/// let a:f64 = div!(f, 0, 0);
/// assert_eq!(a, 4.);
/// ```
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
#[doc(hidden)]
pub fn grad(f:&Function) -> VectorFunction {
    match f {
        Function::OneD(_) => panic!("Gradient vector not defined for 1D function"),
        Function::TwoD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            VectorFunction::TwoD(_VectorFunction2D {
                f1: Function::TwoD(_Function2D {
                    f: Box::new(move |x:f64, y:f64| ddx_s(&f1, vec![x, y])),
                    expression: format!("ddx({})", f.expression())
                }),
                f2: Function::TwoD(_Function2D {
                    f: Box::new(move |x:f64, y:f64| ddy_s(&f2, vec![x, y])),
                    expression: format!("ddy({})", f.expression())
                }),
                potential: Some(f.clone()),
            })
        }
        Function::ThreeD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            let f3 = f.clone();
            VectorFunction::ThreeD(_VectorFunction3D {
                f1: Function::ThreeD(_Function3D {
                    f: Box::new(move |x:f64, y:f64, z:f64| ddx_s(&f1, vec![x, y, z])),
                    expression: format!("ddx({})", f.expression())
                }),
                f2: Function::ThreeD(_Function3D {
                    f: Box::new(move |x:f64, y:f64, z:f64| ddy_s(&f2, vec![x, y, z])),
                    expression: format!("ddy({})", f.expression())
                }),
                f3: Function::ThreeD(_Function3D {
                    f: Box::new(move |x:f64, y:f64, z:f64| ddz_s(&f3, vec![x, y, z])),
                    expression: format!("ddz({})", f.expression())
                }),
                potential: Some(f.clone())
            })
        }
    }
}
/// Gradient for functions
/// # Gradient
/// This macro only takes a [Function] as an argument, and returns a [VectorFunction], whose components
/// are the partial derivatives of the original function. \
/// Vector functions created like this preserve its potential function and you can access it as an expression
/// with `f.potential_expression()`.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f = f!(x, y, x*y);
/// let v = grad!(f);
/// assert_eq!(v(4., 5.), vector!(5.000000000165983, 3.9999999998485687)); // Analytically, v(4, 4)=(5, 4)
/// ```
#[macro_export]
macro_rules! grad {
    ($f:expr) => {
        grad(&$f)
    }
}

// ----- PARAMETRIC CURVES -----
#[doc(hidden)]
#[derive(Clone)]
pub struct _ParametricCurve2D { // Supposed to be 1D
    pub f1:Function,
    pub f2:Function,
}
impl _ParametricCurve2D {
    pub fn ddt(&self, t:f64) -> Vector {
        Vector::TwoD(_Vector2::new(((self.f1)(t + Δ) - (self.f1)(t))/Δ, ((self.f2)(t + Δ) - (self.f2)(t))/Δ))
    }
}
impl FnOnce<(f64,)> for _ParametricCurve2D {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        Vector::TwoD(_Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
impl FnMut<(f64,)> for _ParametricCurve2D {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        Vector::TwoD(_Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
impl Fn<(f64,)> for _ParametricCurve2D {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        Vector::TwoD(_Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
#[doc(hidden)]
#[derive(Clone)]
pub struct _ParametricCurve3D {
    pub f1:Function,
    pub f2:Function,
    pub f3:Function,
}
impl _ParametricCurve3D {
    pub fn ddt(&self, t:f64) -> Vector {
        Vector::ThreeD(_Vector3::new(((self.f1)(t + Δ) - (self.f1)(t))/Δ, ((self.f2)(t + Δ) - (self.f2)(t))/Δ, ((self.f3)(t + Δ) - (self.f3)(t))/Δ))
    }
}
impl FnOnce<(f64,)> for _ParametricCurve3D {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
impl FnMut<(f64,)> for _ParametricCurve3D {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
impl Fn<(f64,)> for _ParametricCurve3D {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
/// Parametric curves in R^2 and R^3
/// # Parametric Curve
/// Parametric curves are vector functions of one variable, typically representing curves in space. \
/// To create one there is the [curve!] macro, although most of the time these will be used in a [Contour],
/// and so they will be created in one as such.\
/// These can be evaluated like single-variable functions, they can be cloned, and you obtain the
/// derivative vector through the `c.ddt(t0)` method or the [ddt!] macro.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let sigma:ParametricCurve = curve!(t, t.powi(2), 3.*t); // σ(t) = (t^2, 3t)
/// assert_eq!(sigma(2.), vector!(4, 6)); // σ(2) = (4, 6)
/// ```
#[derive(Clone)]
pub enum ParametricCurve {
    TwoD(_ParametricCurve2D),
    ThreeD(_ParametricCurve3D)
}
impl ParametricCurve {
    pub fn ddt(&self, t:f64) -> Vector {
        match self {
            ParametricCurve::TwoD(sigma) => sigma.ddt(t),
            ParametricCurve::ThreeD(sigma) => sigma.ddt(t)
        }
    }
}
impl Display for ParametricCurve {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParametricCurve::TwoD(sigma) => {
                write!(f, "⟨{}, {}⟩", sigma.f1.expression(), sigma.f2.expression())
            }
            ParametricCurve::ThreeD(sigma) => {
                write!(f, "⟨{}, {}, {}⟩", sigma.f1.expression(), sigma.f2.expression(), sigma.f3.expression())
            }
        }
    }
}
/// Creates parametric curves
/// # Curve macro
/// This macro takes in a single variable identifier and n expressions involving that variable, where n
/// is either 2 or 3 depending on the dimension of the curve, and returns a [ParametricCurve]
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let rho:ParametricCurve = curve!(tau, 3.*tau, 5., tau.sqrt()); // curve in R^3
/// ```
#[macro_export]
macro_rules! curve {
    ($t:ident, $f1:expr, $f2:expr) => {
        ParametricCurve::TwoD(_ParametricCurve2D {
            f1: Function::OneD(_Function1D {
                f: Box::new(|$t:f64| $f1),
                expression: String::from(stringify!($f1))
            }),
            f2: Function::OneD(_Function1D {
                f: Box::new(|$t:f64| $f2),
                expression: String::from(stringify!($f2))
            })
        })
    };
    ($t:ident, $f1:expr, $f2:expr, $f3:expr) => {
        ParametricCurve::ThreeD(_ParametricCurve3D {
            f1: Function::OneD(_Function1D {
                f: Box::new(|$t:f64| $f1),
                expression: String::from(stringify!($f1))
            }),
            f2: Function::OneD(_Function1D {
                f: Box::new(|$t:f64| $f2),
                expression: String::from(stringify!($f2))
            }),
            f3: Function::OneD(_Function1D {
                f: Box::new(|$t:f64| $f3),
                expression: String::from(stringify!($f3))
            })
        })
    };
}
impl FnOnce<(f64,)> for ParametricCurve {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        match self {
            ParametricCurve::TwoD(sigma) => sigma(args.0),
            ParametricCurve::ThreeD(sigma) => sigma(args.0)
        }
    }
}
impl FnMut<(f64,)> for ParametricCurve {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        match self {
            ParametricCurve::TwoD(sigma) => sigma(args.0),
            ParametricCurve::ThreeD(sigma) => sigma(args.0)
        }
    }
}
impl Fn<(f64,)> for ParametricCurve {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        match self {
            ParametricCurve::TwoD(sigma) => sigma(args.0),
            ParametricCurve::ThreeD(sigma) => sigma(args.0)
        }
    }
}
/// Derivative for parametric curves
/// # Derivative macro
/// This macro is another option to the `sigma.ddt(f64)` method, and takes in a [ParametricCurve] or a [Contour]
/// and a f64, and returns the derivative vector on t0=f64.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let sigma:ParametricCurve = curve!(t, t.powi(2), 3.*t); // σ(t) = (t^2, 3t)
/// assert_eq!(sigma.ddt(1.), vector!(2.000005000013516, 3.000000000064062)) // σ'(1) = (2, 3);
/// ```
#[macro_export]
macro_rules! ddt {
    ($f:expr, $t:expr) => {$f.ddt($t as f64)};
}

// ----- SETS -----
#[doc(hidden)]
pub trait Super {
    fn wrap(&self) -> _SuperSet;
}
impl Super for Set {
    fn wrap(&self) -> _SuperSet {
        _SuperSet::Set(self)
    }
}
impl Super for FSet {
    fn wrap(&self) -> _SuperSet {
        _SuperSet::FSet(self)
    }
}
/// Single variable domains
/// # Set
/// A set in this crate is the domain where a variable lives. It's commonly used next to a [ParametricCurve]
/// in a [Contour] to delimit the bounds of its independent variable. Like t ∈ \[a, b\]. \
/// You can access the start and end of a set as `.i` and `.f` \
/// There's also the `.linspace(n)` method that returns a vector with n elements of the set split n times. \
/// They are created with the [set!] macro, and do implement display.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let t_space:Set = set![0, 1];
/// assert_eq!(t_space.i, 0.0);
/// ```
#[derive(Copy, Clone)]
pub struct Set {
    pub i:f64,
    pub f:f64
}
impl Set {
    pub fn linspace(&self, n:i32) -> Vec<f64> {
        let mut space:Vec<f64> = vec![self.i];
        let δ:f64 = (self.f-self.i)/(n as f64);
        for i in 0..n {
            space.push(self.i+ δ*(i as f64));
        }
        space
    }
}
/// Function sets for variable domains
/// # Function Set
/// An FSet is like a [Set] that instead of constant f64 limits has single-variable functions as its bounds. \
/// It's commonly used for the bounds of one variable in a [ParametricSurface]. \
/// These are created with the [fset!] macro. \
/// _Note:_ These can't be evaluated like functions, and are meant just for parametric surfaces, but they do
/// implement display, so when printed they will print the expressions as strings.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let y_bounds:FSet = fset!(f!(x, 0.), f!(x, x.powi(2)));
/// ```
#[derive(Clone)]
pub struct FSet {
    pub i:Function,
    pub f:Function,
}
impl FSet {
    pub fn new(i:Function, f:Function) -> Self {
        match (&i, &f) {
            (Function::OneD(_), Function::OneD(_)) |
            (Function::TwoD(_), Function::TwoD(_)) => {
                FSet { i, f }
            },
            (Function::ThreeD(_), Function::ThreeD(_)) => panic!("Functions can't be of dimension 3"),
            (_, _) => panic!("Functions must be of the same dimension")
        }
    }
}
#[doc(hidden)]
#[derive(Clone)]
pub enum _SuperSet<'s> {
    Set(&'s Set),
    FSet(&'s FSet)
}
/// Creates a set
/// # Set macro
/// This macro is used to create a [Set], and it takes two f64's as arguments representing its bounds
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let t_space:Set = set![0, std::f64::consts::PI];
/// ```
#[macro_export]
macro_rules! set {
    ($i:expr, $f:expr) => {Set {
        i: $i as f64,
        f: $f as f64
    }};
}
/// Creates a function set
/// # Function set macro
/// This macro is used to create an [FSet], and it takes two [Function]s as arguments, which can be already
/// created, or you can use the [f!] macro to create them on the spot, but they both need to be single-variable.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let y_bounds:FSet = fset!(f!(x, 0.), f!(x, x.powi(2)));
/// ```
#[macro_export]
macro_rules! fset {
    ($fi:expr, $ff:expr) => {FSet::new($fi, $ff)};
}
impl Display for Set {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.5}, {:.5}]", self.i, self.f)
    }
}
impl Display for FSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.i.expression(), self.f.expression())
    }
}

// ----- CONTOURS -----
#[doc(hidden)]
#[derive(Clone)]
pub struct _Contour2D {
    pub f_t: _ParametricCurve2D,
    pub lim: Set
}
#[doc(hidden)]
#[derive(Clone)]
pub struct _Contour3D {
    pub f_t: _ParametricCurve3D,
    pub lim: Set
}
/// Contours for integration
/// # Contour
/// Contours are composed of a [ParametricCurve] and a [Set], which represent the curve and the bounds,
/// due to these being intended for a [line_integral!]. \
/// Contours also have a `.ddt(f64)` method (and the [ddt!] macro), a `.linspace()` and a `.bounds()` method, which returns a tuple
/// with two f64 as the bounds. \
/// These can be created with the [contour!] macro, and can also be evaluated as functions. \
/// Lastly, contours have a `.len()` method which calculates the length of the curve given its bounds.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let c:Contour = contour!(t, 2.*t, t.powi(2), 0, 10);
/// assert_eq!(c(3.), vector!(6, 9));
/// ```
#[derive(Clone)]
pub enum Contour {
    TwoD(_Contour2D),
    ThreeD(_Contour3D)
}
impl Contour {
    pub fn ddt(&self, t:f64) -> Vector {
        match self {
            Contour::TwoD(c) => c.f_t.ddt(t),
            Contour::ThreeD(c) => c.f_t.ddt(t)
        }
    }
    pub fn bounds(&self) -> (f64, f64) {
        match self {
            Contour::TwoD(c) => (c.lim.i, c.lim.f),
            Contour::ThreeD(c) => (c.lim.i, c.lim.f)
        }
    }
    pub fn linspace(&self, t:i32) -> Vec<f64> {
        match self {
            Contour::TwoD(c) => c.lim.linspace(t),
            Contour::ThreeD(c) => c.lim.linspace(t)
        }
    }
    pub fn len(&self) -> f64 {
        let g = self.clone();
        match self {
            Contour::TwoD(c) => {
                let f = Function::OneD(_Function1D {
                    f: Box::new(move |t:f64| !g.ddt(t)),
                    expression: String::from(""),
                });
                integral_1d(&f, &c.lim, IntegrationMethod::GaussLegendre)
            },
            Contour::ThreeD(c) => {
                let f = Function::OneD(_Function1D {
                    f: Box::new(move |t:f64| !g.ddt(t)),
                    expression: String::from(""),
                });
                integral_1d(&f, &c.lim, IntegrationMethod::GaussLegendre)
            }
        }
    }
}
impl Display for Contour {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Contour::TwoD(c) => {
                write!(f, "⟨{}, {}⟩", c.f_t.f1.expression(), c.f_t.f2.expression())
            }
            Contour::ThreeD(c) => {
                write!(f, "⟨{}, {}, {}⟩", c.f_t.f1.expression(), c.f_t.f2.expression(), c.f_t.f3.expression())
            }
        }
    }
}
/// Creates a contour
/// # Contour macro
/// This macro is used to create a [Contour], and can be initialized with a [ParametricCurve] and a [Set],
/// or an identifier coupled with n expressions and two f64's for the bounds.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let p:ParametricCurve = curve!(t, 2.*t, t.powi(2)); // curve
/// let c:Contour = contour!(p, set![0, 5]); // curve + set
/// let rho:Contour = contour!(t, 3.*t, 0., t.sqrt(), 0, 2); // from scratch
/// ```
#[macro_export]
macro_rules! contour {
    ($t:ident, $f1:expr, $f2:expr, $t0:expr, $t1:expr) => {
        Contour::TwoD(_Contour2D {
            f_t: _ParametricCurve2D {
                f1: Function::OneD(_Function1D {
                    f: Box::new(|$t:f64| $f1),
                    expression: String::from(stringify!($f1))
                }),
                f2: Function::OneD(_Function1D {
                    f: Box::new(|$t:f64| $f2),
                    expression: String::from(stringify!($f2))
                })
            },
            lim: set![$t0, $t1]
        })
    };
    ($t:ident, $f1:expr, $f2:expr, $f3:expr, $t0:expr, $t1:expr) => {
        Contour::ThreeD(_Contour3D {
            f_t: _ParametricCurve3D {
                f1: Function::OneD(_Function1D {
                    f: Box::new(|$t:f64| $f1),
                    expression: String::from(stringify!($f1))
                }),
                f2: Function::OneD(_Function1D {
                    f: Box::new(|$t:f64| $f2),
                    expression: String::from(stringify!($f2))
                }),
                f3: Function::OneD(_Function1D {
                    f: Box::new(|$t:f64| $f3),
                    expression: String::from(stringify!($f3))
                })
            },
            lim: set![$t0, $t1]
        })
    };
    ($curve:expr, $set:expr) => {
        match $curve {
            ParametricCurve::TwoD(c) => {
                Contour::TwoD(_Contour2D {
                    f_t: c,
                    lim: $set
                })
            },
            ParametricCurve::ThreeD(c) => {
                Contour::ThreeD(_Contour3D {
                    f_t: c,
                    lim: $set
                })
            },
            _ => panic!("Not a valid parametric curve")
        }
    }
}
impl FnOnce<(f64,)> for Contour {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        match self {
            Contour::TwoD(c) => (c.f_t)(args.0),
            Contour::ThreeD(c) => (c.f_t)(args.0)
        }
    }
}
impl FnMut<(f64,)> for Contour {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        match self {
            Contour::TwoD(c) => (c.f_t)(args.0),
            Contour::ThreeD(c) => (c.f_t)(args.0)
        }
    }
}
impl Fn<(f64,)> for Contour {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        match self {
            Contour::TwoD(c) => (c.f_t)(args.0),
            Contour::ThreeD(c) => (c.f_t)(args.0)
        }
    }
}
// The ddt macro also works for Contours

// ----- LINE INTEGRAL -----
/// Returns the type of any variable
/// # Type of function
/// Returns the type of any variable passed as reference, as a string
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let u:Vector = vector!(0, 1, 0);
/// assert_eq!(type_of(&u), "Vector");
/// ```
pub fn type_of<T>(_: &T) -> &str {
    std::any::type_name::<T>().split("::").collect::<Vec<&str>>().last().unwrap()
}
#[doc(hidden)]
pub enum __NV {
    Number(f64),
    Vector(Vector)
}
#[doc(hidden)]
pub trait NVWrap {
    fn nv_wrap(&self) -> __NV;
}
impl NVWrap for f64 {
    fn nv_wrap(&self) -> __NV {
        __NV::Number(*self)
    }
}
impl NVWrap for Vector {
    fn nv_wrap(&self) -> __NV {
        __NV::Vector(*self)
    }
}
// Number near macro
macro_rules! _near {
    ($a:expr, $b:expr; $e:expr) => {
        $a > $b - $e && $a < $b + $e
    };
    ($a:expr, $b:expr) => {
        $a > $b - 2.*Δ && $a < $b + 2.*Δ
    };
}
#[doc(hidden)]
pub fn near(a:__NV, b:__NV, e:f64) -> bool {
    match (a, b) {
        (__NV::Number(a), __NV::Number(b)) => {
            _near!(a, b; e)
        },
        (__NV::Vector(u), __NV::Vector(v)) => {
            match (u, v) {
                (Vector::TwoD(u), Vector::TwoD(v)) => {
                    _near!(u.x, v.x; e) && _near!(u.y, v.y; e)
                },
                (Vector::ThreeD(u), Vector::ThreeD(v)) => {
                    _near!(u.x, v.x; e) && _near!(u.y, v.y; e) && _near!(u.z, v.z; e)
                },
                (Vector::TwoD(u), Vector::ThreeD(v)) => {
                    _near!(u.x, v.x; e) && _near!(u.y, v.y; e) && _near!(0.0, v.z; e)
                },
                (Vector::ThreeD(u), Vector::TwoD(v)) => {
                    _near!(u.x, v.x; e) && _near!(u.y, v.y; e) && _near!(u.z, 0.0; e)
                },
            }
        },
        (_, _) => panic!("Can only compare numbers with numbers and vectors with vectors")
    }
}
/// Checks if two numbers or vectors are close enough
/// # Near macro
/// The near macro takes two f64's or two [Vector]s -- optionally also an error threshold f64, -- and
/// returns `ture` or `false` depending on if they are close enough. The default error threshold is 2*[Δ]=10^-5.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f:Contour = contour!(t, 2.*t, t.powi(2), 0, 1);
/// let v:Vector = vector!(2, 2);
/// assert!(near!(f.ddt(1.), v)); // f'(1) = (2, 2)
/// ```
#[macro_export]
macro_rules! near {
    ($a:expr, $b:expr) => {
        near($a.nv_wrap(), $b.nv_wrap(), 2.*Δ)
    };
    ($a:expr, $b:expr; $e:expr) => {
        near($a.nv_wrap(), $b.nv_wrap(), $e)
    };
}
// In-crate near vector efficient macro
macro_rules! _near_v {
    ($u:expr, $v:expr) => {
        match ($u, $v) {
            (Vector::TwoD(u), Vector::TwoD(v)) => {
                _near!(u.x, v.x) && _near!(u.y, v.y)
            },
            (Vector::ThreeD(u), Vector::ThreeD(v)) => {
                _near!(u.x, v.x) && _near!(u.y, v.y) && _near!(u.z, v.z)
            },
            (Vector::TwoD(u), Vector::ThreeD(v)) => {
                _near!(u.x, v.x) && _near!(u.y, v.y) && _near!(0.0, v.z)
            },
            (Vector::ThreeD(u), Vector::TwoD(v)) => {
                _near!(u.x, v.x) && _near!(u.y, v.y) && _near!(u.z, 0.0)
            }
        }
    };
    ($u:expr, $v:expr; $e:expr) => {
        match ($u, $v) {
            (Vector::TwoD(u), Vector::TwoD(v)) => {
                _near!(u.x, v.x; $e) && _near!(u.y, v.y; $e)
            },
            (Vector::ThreeD(u), Vector::ThreeD(v)) => {
                _near!(u.x, v.x; $e) && _near!(u.y, v.y; $e) && _near!(u.z, v.z; $e)
            },
            (Vector::TwoD(u), Vector::ThreeD(v)) => {
                _near!(u.x, v.x; $e) && _near!(u.y, v.y; $e) && _near!(0.0, v.z; $e)
            },
            (Vector::ThreeD(u), Vector::TwoD(v)) => {
                _near!(u.x, v.x; $e) && _near!(u.y, v.y; $e) && _near!(u.z, 0.0; $e)
            },
        }
    };
}
// Single Integration
/// Alternate methods for single integration
/// # Integration Method
/// This enum is used in the [integral!] macro, and contains three variants:
/// - GaussLegendre #default
/// - Riemann(i32)
/// - Simpson13(i32)
///
/// The Riemann and Simpson 1/3 methods require a number of partitions n, whilst Gauss-Legendre
/// is done with 5 points.
#[derive(Clone)]
pub enum IntegrationMethod {
    GaussLegendre,
    Riemann(i32),
    Simpson13(i32),
}
macro_rules! int_gauss_legendre {
    ($ft:expr, $t0:expr, $t1:expr) => {{
        let t = ($t0 + $t1)/2.;
        let dt = ($t1 - $t0)/2.;
        let w1 = 128f64/225f64;
        let w2 = (322f64+13f64*70f64.sqrt())/900f64;
        let w3 = (322f64-13f64*70f64.sqrt())/900f64;
        let x2 = (1f64/3f64)*(5f64-2f64*(10f64/7f64).sqrt()).sqrt()*dt;
        let x3 = (1f64/3f64)*(5f64+2f64*(10f64/7f64).sqrt()).sqrt()*dt;
        return w1*$ft(t + 0.0)*dt + w2*($ft(t - x2)*dt+$ft(t + x2)*dt) + w3*($ft(t - x3)*dt+$ft(t + x3)*dt);
    }};
}
macro_rules! int_riemann {
    ($ft:expr, $t0:expr, $t1:expr, $n:expr) => {{
        let δ:f64 = ($t1-$t0)/($n as f64);
        let mut sum:f64 = 0.;
        for i in 0..$n {
            sum += $ft($t0 + δ*i as f64)
        }
        return sum*δ;
    }};
}
macro_rules! int_simpson13 {
    ($ft:expr, $t0:expr, $t1:expr, $n:expr) => {{
        if $n%2 != 0 { panic!("Simpson's n has to be even") }
            let δ:f64 = ($t1-$t0)/($n as f64);
            let mut sum:f64 = 0.;
            let mut xi:f64;
            for i in 1..$n {
                xi = $t0 + i as f64 * δ;
                if i%2 != 0 {
                    sum += 4.0*$ft(xi);
                } else {
                    sum += 2.0*$ft(xi);
                }
            }
            1./3. * δ * ($ft($t0) + $ft($t1) + sum)
    }};
}
// Multiple integration
/// Alternate methods for double/triple integration
/// # Multiple Integration Method
/// This enum is used in the [integral!] macro, and contains three variants:
/// - MonteCarlo(i32) #default
/// - MidPoint(f64)
/// - Simpson(f64)
///
/// MonteCarlo requires a number of points n, but the default is 400. \
/// As of the mid-point rule and the Simpson method, these require a δ; recommended is δ=0.05. \
/// _Note:_ midpoint and simpson are only implemented for double integrals as of the current version.
#[derive(Clone)]
pub enum MultipleIntegrationMethod {
    MonteCarlo(i32),
    MidPoint(f64),
    Simpson(f64)
}
fn int_monte_carlo_2d(f:&Function, a:&_SuperSet, b:&_SuperSet, n:i32) -> f64 {
    if let Function::TwoD(_) = f {
        let mut rng = rand::thread_rng();
        match (a, b) {
            (_SuperSet::Set(a), _SuperSet::Set(b)) => {
                let mut sum = 0.0;
                for _ in 0..n {
                    let x = rng.gen_range(a.i..a.f);
                    let y = rng.gen_range(b.i..b.f);
                    sum += f(x, y);
                }
                return (sum / n as f64) * (a.f - a.i) * (b.f - b.i);
            },
            (_SuperSet::Set(a), _SuperSet::FSet(b)) => {
                if let Function::OneD(_) = b.i {
                    let mut sum = 0.0;
                    let mut var = 0.0;
                    for _ in 0..n {
                        let x = rng.gen_range(a.i..a.f);
                        let (yi, yf) = ((b.i)(x), (b.f)(x));
                        let y = rng.gen_range(yi..yf);
                        sum += f(x, y);
                        var += yf - yi;
                    }
                    return (sum / n as f64)*(a.f - a.i)*(var / n as f64);
                } else { panic!("Function limits for this double integral need to be 1D") }
            },
            (_SuperSet::FSet(a), _SuperSet::Set(b)) => {
                if let Function::OneD(_) = a.i {
                    let mut sum = 0.0;
                    let mut var = 0.0;
                    for _ in 0..n {
                        let y = rng.gen_range(b.i..b.f);
                        let (xi, xf) = ((a.i)(y), (a.f)(y));
                        let x = rng.gen_range(xi..xf);
                        sum += f(x, y);
                        var += xf - xi;
                    }
                    return (sum / (n as f64))*(b.f - b.i)*(var / n as f64);
                } else { panic!("Function limits for this double integral need to be 1D") }
            },
            (_, _) => panic!("Both bounds can't be functions")
        }
    } else { panic!("2D Functions require 2 bounds") }
}
fn int_monte_carlo_3d(f:&Function, a:&_SuperSet, b:&_SuperSet, c:&_SuperSet, n:i32) -> f64 {
    if let Function::ThreeD(_) = f {
        let mut rng = rand::thread_rng();
        match (a, b, c) {
            (_SuperSet::Set(a), _SuperSet::Set(b), _SuperSet::Set(c)) => { // any
                let mut sum = 0.0;
                for _ in 0..n {
                    let x = rng.gen_range(a.i..a.f);
                    let y = rng.gen_range(b.i..b.f);
                    let z = rng.gen_range(c.i..c.f);
                    sum += f(x, y, z);
                }
                return (sum / n as f64)*(a.f-a.i)*(b.f-b.i)*(c.f-c.i)
            },
            (_, _, _) => panic!("Triple integrals with variable bounds not supported yet"),
            /*(SuperSet::Set(a), SuperSet::Set(b), SuperSet::FSet(c)) => { // dz dx dy | dz dy dx
                    if let Function::TwoD(_) = c.i {
                        let mut sum = 0.0;
                        let mut var = 0.0;
                        for _ in 0..n {
                            let x = rng.gen_range(a.i..a.f);
                            let y = rng.gen_range(b.i..b.f);
                            let (zi, zf) = ((c.i)(x, y), (c.f)(x , y));
                            let z = rng.gen_range(zi..zf);
                            sum += f(x, y, z);
                            var += zf - zi;
                        }
                        return (sum / n as f64)*(a.f-a.i)*(b.f-b.i)*(var / n as f64);
                    } else if let Function::OneD(_) = c.i {
                        panic!("Make the dy functions a 2D function and not use the other parameter to specify")
                    } else { panic!("dy has to be a 2D function set")}
                },
                (SuperSet::Set(a), SuperSet::FSet(b), SuperSet::Set(c)) => { // dy dx dz | dy dz dx
                    if let Function::TwoD(_) = b.i {
                        let mut sum = 0.0;
                        let mut var = 0.0;
                        for _ in 0..n {
                            let x = rng.gen_range(a.i..a.f);
                            let z = rng.gen_range(c.i..c.f);
                            let (yi, yf) = ((b.i)(x, z), (b.f)(x , z));
                            let y = rng.gen_range(yi..yf);
                            sum += f(x, y, z);
                            var += yf - yi;
                        }
                        return (sum / n as f64)*(a.f-a.i)*(c.f-c.i)*(var / n as f64);
                    } else if let Function::OneD(_) = b.i {
                        panic!("Make the dz functions a 2D function and not use the other parameter to specify")
                    } else { panic!("dz has to be a 2D function set")}
                },
                (SuperSet::FSet(a), SuperSet::Set(b), SuperSet::Set(c)) => { // dx dy dz | dx dz dy
                    if let Function::TwoD(_) = a.i {
                        let mut sum = 0.0;
                        let mut var = 0.0;
                        for _ in 0..n {
                            let x = rng.gen_range(b.i..b.f);
                            let z = rng.gen_range(c.i..c.f);
                            let (xi, xf) = ((a.i)(x, z), (a.f)(x , z));
                            let y = rng.gen_range(xi..xf);
                            sum += f(x, y, z);
                            var += xf - xi;
                        }
                        return (sum / n as f64)*(b.f-b.i)*(c.f-c.i)*(var / n as f64);
                    } else if let Function::OneD(_) = a.i {
                        panic!("Make the dx functions a 2D function and not use the other parameter to specify")
                    } else { panic!("dx has to be a 2D function set")}
                },
                (SuperSet::Set(a), SuperSet::FSet(b), SuperSet::FSet(c)) => { // dz dx dy | dz dy dx
                    if let Function::TwoD(_) = c.i {
                        if let Function::OneD(_) = b.i { // Meaning z is inner and y is outer
                            let mut sum = 0.0;
                            let mut varb = 0.0;
                            let mut varc = 0.0;
                            for _ in 0..n {
                                let x = rng.gen_range(a.i..a.f);
                                let (yi, yf) = ((b.i)(x), (b.f)(x));
                                let y = rng.gen_range(yi..yf);
                                let (zi, zf) = ((c.i)(x,y), (c.f)(x,y));
                                let z = rng.gen_range(zi..zf);
                                sum += f(x,y,z);
                                varb += yf - yi;
                                varc += zf - zi;
                            }
                            return (sum / n as f64)*(a.f-a.i)*(varb / n as f64)*(varc / n as f64);
                        } else if let Function::TwoD(_) = b.i { panic!("Both bound functions can't be 2D") } else {panic!("Fak1")}
                    } else if let Function::OneD(_) = b.i {
                        if let Function::TwoD(_) = c.i { // Meaning y is inner and z is outer
                            let mut sum = 0.0;
                            let mut varb = 0.0;
                            let mut varc = 0.0;
                            for _ in 0..n {
                                let x = rng.gen_range(a.i..a.f);
                                let (zi, zf) = ((c.i)(x), (c.f)(x));
                                let z = rng.gen_range(zi..zf);
                                let (yi, yf) = ((b.i)(x,z), (b.f)(x,z));
                                let y = rng.gen_range(yi..yf);
                                sum += f(x,y,z);
                                varb += yf - yi;
                                varc += zf - zi;
                            }
                            return (sum / n as f64)*(a.f-a.i)*(varb / n as f64)*(varc / n as f64);
                        } else if let Function::OneD(_) = c.i { panic!("Use a 2D function and a 1D function, with the 2D being the inner function, even if you don't use one of the variables")}  else {panic!("Fak2")}
                    } else {panic!("One function needs to be 2D and one 1D")}
                },
                (_, _, _) => panic!("All three integral limits can't be functions")*/
        }
    } else { panic!("3D functions require 3 bounds") }
}
macro_rules! int_montecarlo {
    ($f:expr, $a:expr, $b:expr, $n:expr) => {int_monte_carlo_2d(&$f, $a, $b, $n)};
    ($f:expr, $a:expr, $b:expr, $c:expr, $n:expr) => {int_monte_carlo_3d(&$f, $a, $b, $c, $n)};
}
fn int_midpint_2d(f:&Function, a:&_SuperSet, b:&_SuperSet, h:f64) -> f64 {
    if let Function::TwoD(_) = f {
        let mut sum = 0.0;
        match (a, b) {
            (_SuperSet::Set(a), _SuperSet::Set(b)) => {
                let mut x = a.i + 0.5*h;
                let mut y;
                while x < a.f {
                    y = b.i + 0.5*h;
                    while y < b.f {
                        sum += f(x, y)*h.powi(2);
                        y += h;
                    }
                    x += h;
                }
                sum
            },
            (_SuperSet::Set(a), _SuperSet::FSet(b)) => {
                let mut x = a.i + 0.5*h;
                let mut y;
                while x < a.f {
                    y = (b.i)(x) + 0.5*h;
                    while y < (b.f)(x) {
                        sum += f(x, y)*h.powi(2);
                        y += h;
                    }
                    x += h;
                }
                sum
            },
            (_SuperSet::FSet(a), _SuperSet::Set(b)) => {
                let mut y = b.i + 0.5*h;
                let mut x;
                while y < b.f {
                    x = (a.i)(y) + 0.5*h;
                    while y < (a.f)(x) {
                        sum += f(x, y)*h.powi(2);
                        x += h;
                    }
                    y += h;
                }
                sum
            },
            (_, _) => panic!("Both bounds can't be functions")
        }
    } else { panic!("2D Functions require 2 bounds") }
}
fn int_simpson_2d(f:&Function, a:&_SuperSet, b:&_SuperSet, h:f64) -> f64 {
    if let Function::TwoD(_) = f {
        let mut sum = 0.0;
        match (a, b) {
            (_SuperSet::Set(a), _SuperSet::Set(b)) => {
                let mut x = a.i;
                let mut y;
                while x < a.f {
                    y = b.i + 0.5*h;
                    while y < b.f {
                        let mdpt_vol = f(x+0.5*h, y+0.5*h)*h.powi(2);
                        let trap_vol = 0.25*h.powi(2)*(f(x, y) + f(x+h, y) + f(x, y+h) + f(x+h, y+h));
                        sum += (2.*mdpt_vol+trap_vol)/3.;
                        y += h;
                    }
                    x += h;
                }
                sum
            },
            (_SuperSet::Set(a), _SuperSet::FSet(b)) => {
                let mut x = a.i;
                let mut y = 0.0;
                while x < a.f {
                    y = (b.i)(x) + 0.5*h;
                    while y < (b.f)(x) {
                        let mdpt_vol = f(x+0.5*h, y+0.5*h)*h.powi(2);
                        let trap_vol = 0.25*h.powi(2)*(f(x, y) + f(x+h, y) + f(x, y+h) + f(x+h, y+h));
                        sum += (2.*mdpt_vol+trap_vol)/3.;
                        y += h;
                    }
                    x += h;
                }
                sum
            },
            (_SuperSet::FSet(a), _SuperSet::Set(b)) => {
                let mut y = b.i;
                let mut x = 0.0;
                while y < b.f {
                    x = (a.i)(x) + 0.5*h;
                    while x < (a.f)(y) {
                        let mdpt_vol = f(x+0.5*h, y+0.5*h)*h.powi(2);
                        let trap_vol = 0.25*h.powi(2)*(f(x, y) + f(x+h, y) + f(x, y+h) + f(x+h, y+h));
                        sum += (2.*mdpt_vol+trap_vol)/3.;
                        x += h;
                    }
                    y += h;
                }
                sum
            },
            (_, _) => panic!("Both bounds can't be functions")
        }
    } else { panic!("2D Functions require 2 bounds") }
}
// General Function Wrapper
#[doc(hidden)]
pub enum __G<'s> {
    Function(&'s Function),
    VectorFunction(&'s VectorFunction)
}
#[doc(hidden)]
pub trait GWrap {
    fn wrap(&self) -> __G;
} // Wrapper for General Function
impl GWrap for Function {
    fn wrap(&self) -> __G {
        __G::Function(self)
    }
}
impl GWrap for VectorFunction {
    fn wrap(&self) -> __G {
        __G::VectorFunction(&self)
    }
}
/// Line integrals for scalar and vector functions
/// # Line Integral macro
/// Line integrals can take a [Function] or [VectorFunction], along with a [Contour] and optionally
/// an [IntegrationMethod].
/// ## Examples
/// ```
/// use vector_calculus::*;
/// setup!();
/// let f:Function = f!(x, y, x + y);
/// let F:VectorFunction = vector_function!(x, y, x*y, y.powi(2));
/// let c:Contour = contour!(t, cos!(t), sin!(t), 0, 2.*PI);
/// // Scalar line integral
/// assert_eq!(line_integral!(f, c), -0.00019354285449835196); // Analytically its 0
/// // Vector line integral              v 3e-12 = 0
/// assert_eq!(line_integral!(F, c), -3.877467549578482e-12); // Analytically its 0
/// ```
#[macro_export]
macro_rules! line_integral {
    ($g:expr, $c:expr) => {
        line_integral($g.wrap(), &$c, IntegrationMethod::GaussLegendre)
    };
    ($g:expr, $c:expr, $m:expr) => {
        line_integral($g.wrap(), &$c, $m)
    };
}
#[doc(hidden)]
pub fn line_integral(g: __G, c:&Contour, method:IntegrationMethod) -> f64 {
    let (t0, t1) = c.bounds();
    let mut ft:Box<dyn Fn(f64)->f64> = Box::new(|_:f64| f64::NAN);
    match g {
        __G::Function(f) => {
            match (f, c) {
                (Function::TwoD(_), Contour::TwoD(_)) |
                (Function::ThreeD(_), Contour::ThreeD(_)) |
                (Function::ThreeD(_), Contour::TwoD(_)) => {
                    ft = Box::new(move |t:f64| {
                        f(c(t))*!ddt!(c, t)
                    });
                },
                (Function::OneD(_), _) => panic!("For 1D functions, use the integral! macro"),
                _ => panic!("No line integral of a 2D function over a 3D contour")
            }
        }
        __G::VectorFunction(v) => {
            let (c1,c0) = (c(t1), c(t0));
            let (half, third) = ((71./67.)*((t1-t0)/2.), (129./131.)*((t1-t0)/3.));
            match v {
                VectorFunction::TwoD(vf) => {
                    match c {
                        Contour::TwoD(_) => {
                            if let Some(f) = vf.potential.clone() {
                                if _near_v!(c1, c0; Δ*1e-7) {
                                    return 0.
                                } else {
                                    return f(c1) - f(c0);
                                }
                            } else {
                                let (half, third) = (!curl!(v, c(half).x(), c(half).y()), !curl!(v, c(third).x(), c(third).y()));
                                if near!(half, 0.0) && near!(third, 0.0) && _near_v!(c1, c0){
                                    return 0.
                                } else {
                                    ft = Box::new(move |t:f64| {
                                        return v(c(t))*ddt!(c, t)
                                    });
                                }
                            }
                        },
                        Contour::ThreeD(_) => panic!("Line integral of a 3D contour on a 2D vector function")
                    }
                }
                VectorFunction::ThreeD(vf) => {
                    if let Some(f) = vf.potential.clone() {
                        if _near_v!(c1, c0; Δ*1e-7) {
                            return 0.
                        } else {
                            return f(c1) - f(c0);
                        }
                    }
                    match c {
                        Contour::TwoD(_) => {
                            let (half, third) = (!curl!(v, c(half).x(), c(half).y(), 0.), !curl!(v, c(third).x(), c(third).y(), 0.));
                            if near!(half, 0.0) && near!(third, 0.0) && _near_v!(c1, c0){
                                return 0.
                            } else {
                                ft = Box::new(move |t:f64| {
                                    return v(c(t))*ddt!(c, t)
                                });
                            }
                        }
                        Contour::ThreeD(_) => {
                            let (half, third) = (!curl!(v, c(half).x(), c(half).y(), c(half).z()), !curl!(v, c(third).x(), c(third).y(), c(third).z()));
                            if near!(half, 0.0) && near!(third, 0.0) && _near_v!(c1, c0){
                                return 0.
                            } else {
                                ft = Box::new(move |t:f64| {
                                    return v(c(t))*ddt!(c, t)
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    return match method {
        IntegrationMethod::GaussLegendre => int_gauss_legendre!(ft, t0, t1),
        IntegrationMethod::Riemann(n) => int_riemann!(ft, t0, t1, n),
        IntegrationMethod::Simpson13(n) => int_simpson13!(ft, t0, t1, n)
    }
}
#[doc(hidden)]
pub fn integral_1d(f:&Function, set:&Set, method:IntegrationMethod) -> f64 {
    match f {
        Function::OneD(_) => match method {
            IntegrationMethod::GaussLegendre => int_gauss_legendre!(f, set.i, set.f),
            IntegrationMethod::Riemann(n) => int_riemann!(f, set.i, set.f, n),
            IntegrationMethod::Simpson13(n) => int_simpson13!(f, set.i, set.f, n)
        },
        _ => panic!("For 2D and 3D scalar functions use the line_integral! macro")
    }
}

// ----- DOUBLE/TRIPLE INTEGRAL -----
#[doc(hidden)]
pub fn rn_integral(f:&Function, lim:Vec<_SuperSet>, method:MultipleIntegrationMethod) -> f64 {
    match (f, lim.len()) {
        (Function::TwoD(_), 2) => {
            match method {
                MultipleIntegrationMethod::MonteCarlo(n) => int_monte_carlo_2d(f, &lim[0], &lim[1], n),
                MultipleIntegrationMethod::MidPoint(h) => int_midpint_2d(f, &lim[0], &lim[1], h),
                MultipleIntegrationMethod::Simpson(h) => int_simpson_2d(f, &lim[0], &lim[1], h)
            }
        },
        (Function::ThreeD(_), 3) => {
            match (&lim[0], &lim[1], &lim[2]) {
                (_SuperSet::Set(_), _SuperSet::Set(_), _SuperSet::Set(_)) => {
                    match method {
                        MultipleIntegrationMethod::MonteCarlo(n) => int_monte_carlo_3d(f, &lim[0], &lim[1], &lim[2], n),
                        MultipleIntegrationMethod::MidPoint(_) => panic!("No mid point method yet for 3D functions"),
                        MultipleIntegrationMethod::Simpson(_) => f64::NAN
                    }
                },
                (_, _, _) => panic!("Triple integration with non-constant bounds is not available as of the current version")
            }
        },
        (_, _) => panic!("Need 3 bounds for 3D functions and 2 bounds for 2D functions")
    }
}
// Integration methods wrapper
#[doc(hidden)]
pub enum __M {
    Single(IntegrationMethod),
    Multi(MultipleIntegrationMethod)
}
// Integration arguments wrapper
#[doc(hidden)]
pub enum __W<'s> {
    Number(f64),
    SSet(_SuperSet<'s>),
    Method(__M),
}
#[doc(hidden)]
pub trait WWrap {
    fn wwrap(&self) -> __W;
}
impl WWrap for f64 {
    fn wwrap(&self) -> __W {
        __W::Number(*self)
    }
}
impl WWrap for Set {
    fn wwrap(&self) -> __W {
        __W::SSet(self.wrap())
    }
}
impl WWrap for FSet {
    fn wwrap(&self) -> __W {
        __W::SSet(self.wrap())
    }
}
impl WWrap for IntegrationMethod {
    fn wwrap(&self) -> __W {
        __W::Method(__M::Single(self.clone()))
    }
}
impl WWrap for MultipleIntegrationMethod {
    fn wwrap(&self) -> __W {
        __W::Method(__M::Multi(self.clone()))
    }
}
#[doc(hidden)]
pub fn categorize_integrals(f:&Function, args:Vec<__W>) -> f64 {
    match args.len() {
        1 => { // 1D set default
            match &args[0] {
                __W::SSet(s) => if let _SuperSet::Set(s) = s {
                    return integral_1d(f, s, IntegrationMethod::GaussLegendre);
                } else { panic!("E071") }
                _ => panic!("E072")
            }
        },
        2 => { // 1D num default | 1D set specify | 2D sets default
            match (&args[0], &args[1]) {
                (__W::Number(a), __W::Number(b)) => {
                    return integral_1d(f, &set![*a, *b], IntegrationMethod::GaussLegendre);
                },
                (__W::SSet(s), __W::Method(m)) => {
                    if let _SuperSet::Set(s) = s {
                        if let __M::Single(m) = m {
                            return integral_1d(f, *s, m.clone());
                        } else { panic!("1D functions can only take methods from the IntegrationMethod enum") }
                    } else { panic!("E074") }
                },
                (__W::SSet(x), __W::SSet(y)) => {
                    return rn_integral(f, vec![x.clone(), y.clone()], MultipleIntegrationMethod::MonteCarlo(400));
                }
                (_, _) => panic!("E075")
            }
        },
        3 => { // 2D sets specify | 3D sets default
            match (&args[0], &args[1], &args[2]) {
                (__W::SSet(x), __W::SSet(y), __W::Method(m)) => {
                    if let __M::Multi(m) = m {
                        return rn_integral(f, vec![x.clone(), y.clone()], m.clone());
                    } else { panic!("2D functions can only take methods from the MultipleIntegrationMethod enum") }
                },
                (__W::SSet(x), __W::SSet(y), __W::SSet(z)) => {
                    return rn_integral(f, vec![x.clone(), y.clone(), z.clone()], MultipleIntegrationMethod::MonteCarlo(400));
                },
                (_, _, _) => panic!("E076")
            }
        },
        4 => { // 3D sets specify
            match (&args[0], &args[1], &args[2], &args[3]) {
                (__W::SSet(x), __W::SSet(y), __W::SSet(z), __W::Method(m)) => {
                    if let __M::Multi(m) = m {
                        return rn_integral(f, vec![x.clone().clone(), y.clone().clone(), z.clone().clone()], m.clone());
                    } else { panic!("3D functions can only take methods from the MultipleIntegrationMethod enum") }
                },
                (_, _, _, _) => panic!("E077")
            }
        },
        _ => panic!("Why you put so many arguments in here, bro?")
    }
}
/// Integrates scalar functions of one, two, or three variables
/// # Integral macro
/// The integral macro is the most versatile of these, because it has multiple ways to call it:
/// 1. With a 1d function and a [Set]
/// 2. With a 1d function and two numbers
/// 3. With a 1d function and a set, specifying the method from the [IntegrationMethod] enum
/// 4. With a 2d function and two sets
/// 5. With a 2d function, two sets and a method from the [MultipleIntegrationMethod] enum
/// 6. With a 3d function and three sets
/// 7. With a 3d function, three sets and the MonteCarlo method from the [MultipleIntegrationMethod] enum,
/// as it is currently the only one supported for triple integrals, but this allows you to specify n: the
/// number of points.
///
/// 2D [Function]s can be integrated with non-constant bounds, where the first set or fset is the x bounds
/// and the second one is the y bounds. However, only one bound can be an [FSet]. \
/// 3D Functions, however, can only be integrated with constant bounds as of the current version; the first
/// set is the x bounds, the second the y bounds and the third the z bounds. \
/// In this implementation, the order of the sets for the integration limits is the same as the definition
/// of the variables. Can't be changed. \
/// This means, that this macro encapsulates, single, double and triple integrals, but not [line_integral!]s
/// nor [surface_integral!]s. \
/// _Note_: It is customary that when the bounds of a variable are functions, you use the other variable as
/// identifier in the FSet functions, as observed in the example with `x_bounds` and `y_bounds`. \
/// _Note_: So you don't have to always write `MultipleIntegrationMethod::Simpson(0.05)`, for example, there's
/// a macro called [setup!], that automatically brings into scope the [IntegrationMethod] and [MultipleIntegrationMethod]
/// enums, so you can just write `Simpson(0.05)` instead.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let f1 = f!(x, x.powi(2));
/// let f2 = f!(x, y, x*y);
/// let f3 = f!(x, y, z, x*y + z);
///
/// let (a, b): (f64, f64) = (0., 1.);
/// let bounds = set![0, 1];
/// let x_bounds = set![0, 1];
/// let y_bounds = fset![f!(x, 0.), f!(x, x.powi(2))];
/// let y_bounds_const = set![0, 3];
/// let z_bounds = set![0, 2];
///
/// let method = IntegrationMethod::Simpson13(100);
/// let multiple_method = MultipleIntegrationMethod::Simpson(0.05);
/// let multiple_method_3d = MultipleIntegrationMethod::MonteCarlo(2_000);
///
/// integral!(f1, bounds); // 1D function
/// integral!(f1, a, b); // 1D function
/// integral!(f1, bounds, method); // 1D function
/// integral!(f2, x_bounds, y_bounds); // 2D function
/// integral!(f2, x_bounds, y_bounds, multiple_method); // 2D function
/// integral!(f3, x_bounds, y_bounds_const, z_bounds); // 3D function
/// integral!(f3, x_bounds, y_bounds_const, z_bounds, multiple_method_3d); // 3D function
/// ```
#[macro_export]
macro_rules! integral {
    ($f:expr, $s:expr) => { // 1D set default
        categorize_integrals(&$f, vec![$s.wwrap()])
    };
    ($f:expr, $asx:expr, $bmy:expr) => { // 1D num default | 1D set specify | 2D sets default
        categorize_integrals(&$f, vec![$asx.wwrap(), $bmy.wwrap()])
    };
    ($f:expr, $x:expr, $y:expr, $mz:expr) => { // 2D sets specify | 3D sets default
        categorize_integrals(&$f, vec![$x.wwrap(), $y.wwrap(), $mz.wwrap()])
    };
    ($f:expr, $x:expr, $y:expr, $z:expr, $m:expr) => { // 3D specified
        categorize_integrals(&$f, vec![$x.wwrap(), $y.wwrap(), $z.wwrap(), $m.wwrap()])
    };
}

// ----- SURFACES -----
/// Special function of R^2→R^3
/// # Parametric Surface
/// Parametric surfaces are part of a [Surface], but you can create one by itself with the [parametric_surface!]
/// macro. They are composed of three scalar two-dimensional functions, and they behave like 2D functions, except
/// they return a 3D vector. \
/// These also implement display, and have a `.ddu(f64,f64)` and `.ddv(f64,f64)` that returns the partial derivative
/// vectors.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let p:ParametricSurface = parametric_surface!(u, v, u.powi(2)*v, u+v, v*u);
/// println!("p(u,v) = {}", p);
/// assert_eq!(p(1., 2.), vector!(2, 3, 2));
/// ```
#[derive(Clone)]
pub struct ParametricSurface { // All supposed to be Function2D
    pub f1:Function,
    pub f2:Function,
    pub f3:Function,
}
impl Display for ParametricSurface {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "⟨{}, {}, {}⟩", self.f1.expression(), self.f2.expression(), self.f3.expression())
    }
}
/// Surfaces in R^3
/// # Surface
/// Surfaces consist of a [ParametricSurface] and two [Set]s, (one of them can be an [FSet]), that represent
/// the bounds of the surfaces. \
/// They are created with the [surface!] macro, and behave just like parametric surfaces, having the same methods
/// but with one more called `.area()` which takes the double integral of the surface to obtain its area.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// setup!();
/// let s:Surface = surface!(u, v, cos!(v)*sin!(u), sin!(v)*sin!(u), cos!(u), 0, PI, 0, 2.*PI); // Sphere
/// assert_eq!(s(0., 0.), vector!(0, 0, 1)); // Top of the sphere
/// ```
#[derive(Clone)]
pub struct Surface<'s> {
    pub f:ParametricSurface,
    pub u_lim: _SuperSet<'s>,
    pub v_lim: _SuperSet<'s>
}
impl ParametricSurface {
    pub fn ddu(&self, u:f64, v:f64) -> Vector {
        Vector::ThreeD(_Vector3::new(((self.f1)(u + Δ, v) - (self.f1)(u, v))/Δ, ((self.f2)(u + Δ, v) - (self.f2)(u, v))/Δ, ((self.f3)(u + Δ, v) - (self.f3)(u, v))/Δ))
    }
    pub fn ddv(&self, u:f64, v:f64) -> Vector {
        Vector::ThreeD(_Vector3::new(((self.f1)(u, v + Δ) - (self.f1)(u, v))/Δ, ((self.f2)(u, v + Δ) - (self.f2)(u, v))/Δ, ((self.f3)(u, v + Δ) - (self.f3)(u, v))/Δ))
    }
}
impl<'s> Surface<'s> {
    pub fn ddu(&self, u:f64, v:f64) -> Vector {
        self.f.ddu(u, v)
    }
    pub fn ddv(&self, u:f64, v:f64) -> Vector {
        self.f.ddv(u, v)
    }
    pub fn area(&self) -> f64 {
        surface_integral(&f!(x, y, z, 1.).wrap(), &self, MultipleIntegrationMethod::MonteCarlo(400))
    }
}
impl FnOnce<(f64, f64)> for ParametricSurface {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl FnMut<(f64, f64)> for ParametricSurface {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl Fn<(f64, f64)> for ParametricSurface {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(_Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl<'s> FnOnce<(f64, f64)> for Surface<'s> {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        (self.f)(args.0, args.1)
    }
}
impl<'s> FnMut<(f64, f64)> for Surface<'s> {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        (self.f)(args.0, args.1)
    }
}
impl<'s> Fn<(f64, f64)> for Surface<'s> {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        (self.f)(args.0, args.1)
    }
}
/// Creates a parametric surface
/// # Parametric Surface macro
/// This macro just takes as input two identifiers (the name of the variables) and three expressions, which
/// will serve as the functions for each component in the 3D vector that it will return when evaluated,
/// because this macro creates a [ParametricSurface].
/// ## Examples
/// ```
/// use vector_calculus::*;
/// let p:ParametricSurface = parametric_surface!(u, v, 3.*u*v, u+v, 9.);
/// ```
#[macro_export]
macro_rules! parametric_surface {
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr) => {
        ParametricSurface {
                f1: f!($u, $v, $f1),
                f2: f!($u, $v, $f2),
                f3: f!($u, $v, $f3),
            }
    };
}
/// Creates a surface
/// # Surface macro
/// This macro is used to initialize a [Surface], and there are multiple ways to call it
/// 1. If all the bounds are constant, you can pass these as f64's.
/// 2. If not all bounds are constant, or you prefer set notation you can do so using a [Set] or [FSet]
/// 3. If you already have a parametric surface initialized previously and would like to turn it into a surface
/// ## Examples
/// ```
/// use vector_calculus::*;
/// // 1. Order of bounds: lower u, higher u, lower v, higher v
/// let s1:Surface = surface!(u, v, u*v, u+v, u.powf(v), 0, 4, 0, 5);
/// // 2. Order of bounds: limits for u, limits for v
/// let s2:Surface = surface!(u, v, u*v, u+v, u.powf(v), set![0, 4], fset![f!(u, 0.), f!(u, 2.*u)]);
/// // 3.
/// let p:ParametricSurface = parametric_surface!(u, v, u*v, u+v, u.powf(v));
/// let s3:Surface = surface!(p, set![0, 4], set![0,5]);
/// ```
#[macro_export]
macro_rules! surface {
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr, $ui:expr, $uf:expr, $vi:expr, $vf:expr) => {
        Surface {
            f: ParametricSurface {
                f1: f!($u, $v, $f1),
                f2: f!($u, $v, $f2),
                f3: f!($u, $v, $f3),
            },
            u_lim: set![$ui, $uf].wrap(),
            v_lim: set![$vi, $vf].wrap()
        }
    };
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr, $ul:expr, $vl:expr) => {
        Surface {
            f: ParametricSurface {
                f1: f!($u, $v, $f1),
                f2: f!($u, $v, $f2),
                f3: f!($u, $v, $f3),
            },
            u_lim: $ul.wrap(),
            v_lim: $vl.wrap()
        }
    };
    ($p:expr, $ul:expr, $vl:expr) => {
        Surface {
            f: $p,
            u_lim: $ul.wrap(),
            v_lim: $vl.wrap()
        }
    }
}

// ----- SURFACE INTEGRAL -----
#[doc(hidden)]
pub fn surface_integral(g:&__G, s:&Surface, m:MultipleIntegrationMethod) -> f64 {
    let s_clone = s.clone();
    let (u, v): (_SuperSet, _SuperSet) = (s_clone.u_lim, s_clone.v_lim);
    let s = s_clone.f;
    match g {
        __G::Function(f) => {
            if let Function::ThreeD(_) = f {
                let fuv = Function::TwoD(_Function2D {
                    f: Box::new(move |u:f64, v:f64| {
                        !(s.ddu(u, v)%s.ddv(u, v))
                    }),
                    expression: String::from("")
                });
                rn_integral(&fuv, vec![u, v], m)
            } else { panic!("No surface integrals for 1D and 2D functions") }
        }
        __G::VectorFunction(f) => {
            let f = f.clone().clone();
            if let VectorFunction::ThreeD(_) = f {
                let vuv = Function::TwoD(_Function2D {
                    f: Box::new(move |u:f64, v:f64| {
                        f(s(u, v))*(s.ddu(u, v)%s.ddv(u,v))
                    }),
                    expression: String::from("")
                });
                rn_integral(&vuv, vec![u, v], m)
            } else { panic!("No surface integrals for 1D and 2D vector functions") }
        }
    }
}
/// Integrates a function (scalar or vector) over a surface
/// # Surface Integral macro
/// This macro is similar to the [line_integral!] macro because it works over scalar or vector functions,
/// but is designed to integrate these over a [Surface]. \
/// You can call this macro with a [Function] or [VectorFunction] and a [Surface], and optionally add in
/// a variant from the [MultipleIntegrationMethod] enum. \
/// By default, it uses the Monte Carlo method with 400, which means its fairly accurate, but it also will
/// produce a slightly different result every time due to the randomness of the method.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// setup!();
/// let F:VectorFunction = vector_function!(x, y, z, x, y, z);
/// let s:Surface = surface!(u, v, cos!(v)*sin!(u), sin!(v)*sin!(u), cos!(u), 0, PI/2., 0, 2.*PI);
/// assert!(near!(surface_integral!(F, s), 2.*PI; 0.2)); // Near to a decimal point of 0.2
/// ```
#[macro_export]
macro_rules! surface_integral {
    ($f:expr, $s:expr) => { surface_integral(&$f.wrap(), &$s, MultipleIntegrationMethod::MonteCarlo(400)) };
    ($f:expr, $s:expr, $m:expr) => { surface_integral(&$f.wrap(), &$s, $m) };
}

// ----- CONFIG -----
/// Initializes the Integration enums as well as pi and e
/// # Setup macro
/// This macro aids by automatically bringing into scope the [IntegrationMethod] and [MultipleIntegrationMethod]
/// enum variants, so you can just write the variant when specifying a method on the [integral!], [line_integral!] or
/// [surface_integral!] macros. This way, just write `Simpson13(100)` instead of `IntegrationMethod::Simpson13(100)`. \
/// This macro also automatically imports the constants pi and e from the `std` library, so it's recommended to
/// use it at the begging of the programs for simplicity.
/// ## Examples
/// ```
/// use vector_calculus::*;
/// setup!();
/// assert!(near!(PI, 3.141592))
/// ```
#[macro_export]
macro_rules! setup {
    () => {
        use $crate::{IntegrationMethod::{self, *}, MultipleIntegrationMethod::{self, *}};
        use std::f64::consts::{PI, E};
    };
}

// ----- HELPERS ------
/// sin function
/// # Sine
/// Just syntax sugar because I think it's easier to write `sin!(x)` than `x.sin()` and it leads to less confusion.
#[macro_export]
macro_rules! sin {
    ($x:expr) => {$x.sin()};
}
/// cos function
/// # Cosine
/// Just syntax sugar because I think it's easier to write `cos!(x)` than `x.cos()` and it leads to less confusion.
#[macro_export]
macro_rules! cos {
    ($x:expr) => {$x.cos()};
}
/// tan function
/// # Tangent
/// Just syntax sugar because I think it's easier to write `tan!(x)` than `x.tan()` and it leads to less confusion.
#[macro_export]
macro_rules! tan {
    ($x:expr) => {$x.tan()};
}
/// ln function
/// # Natural log
/// Just syntax sugar because I think it's easier to write `ln!(x)` than `x.ln()` and it leads to less confusion.
#[macro_export]
macro_rules! ln {
    ($x:expr) => {$x.ln()};
}


use std::f64::consts::{PI, E};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vectors() {
        let u = vector!(3, 4);
        let v = vector!(4, 3);
        let w = u%v;
        println!("|2*(u%v)| = {}, {}", md!(2*w), u.z());
        println!("{}", u);
        //assert_eq!(vector!(0.0, 0.0), vector!(-0.0, -0.0));
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
        assert_eq!(limit!(f => 2, 3), 12.00000000005);
        assert_eq!(f(vector!(2, 3)), 12.);
        assert_eq!(g.expression(), String::from("x.powi(2) * y + z"));

        let h = f!(x, 1./x);
        assert!(near!(integral!(h, set![1., E]), 1.));
        assert!(near!(ddx!(h, 2.), -1./4.));
    }
    //noinspection ALL //This is for the squiggly lines when evaluating Vector Functions
    #[test]
    fn vector_functions() {
        let F:VectorFunction = vector_function!(x, y, -y, x);
        println!("F(1,2) = {:.5}", F(1., 2.));
        println!("∂F/∂y = {}", ddyv!(F, 1, 2));
        println!("|∇xF(1, 2)| = {:.5}", !curl!(F, 1, 2));

        let g = f!(x, y, z, x + y +z);
        let del_g = grad!(g);
        println!("∇g(1, 2, 3) = {}", del_g(1., 2., 3.));
        assert_eq!(del_g.potential(vec![1., 2., 3.]), g(1., 2., 3.));
    }
    #[test]
    fn contours() {
        let sigma:ParametricCurve = curve!(t, t.powi(2), 2.*t);
        let space:Set = set![0., 2.*PI];
        //println!("sigma = {}, space = {}", sigma, space);

        let c:Contour = contour!(t, t.cos(), t.sin(), 0, PI);
        assert!(near!(c.len(), PI));

        println!("c(PI/2) = {}, dcdt(PI) = {}, where t is in {:?}", c(PI/2.), ddt!(c, PI), c.bounds());

        let s = contour!(sigma, space);
        assert_eq!(s(1.), vector!(1, 2));
    }
    #[test]
    fn line_integrals() {
        let g = vector_function!(x, y, 2.*x*y.cos(), -x.powi(2)*y.sin());
        let sigma = contour!(t, (t-1.).exp(), (PI/t).sin(), 1, 2);
        assert!(near!(line_integral!(g, sigma, IntegrationMethod::Simpson13(400)), E.powi(2)*1.0_f64.cos()-1.; 1e-4));

        let f = f!(x, y, x.powi(2)*y);
        let c = contour!(t, t.cos(), t.sin(), 0, PI/2.);
        assert!(near!(line_integral!(f, c), 1./3.))
    }
    #[test]
    fn rn_integrals() {
        let f = f!(x, y, (x.powi(4)+1.).sqrt());
        let x_bounds = fset![f!(y, y.powf(1./3.)), f!(y, 2.)];
        let y_bounds = set![0, 8];
        let a:f64 = integral!(f, x_bounds, y_bounds, MultipleIntegrationMethod::MonteCarlo(1_000));
        assert!(near!(a, (1./6.)*(17_f64.powf(1.5)-1.); 2.2));

        assert!(near!(integral!(f!(x, y, z, x*y + z), set![0, 2], set![0, 3], set![0, 1], MultipleIntegrationMethod::MonteCarlo(2_000)), 12.; 0.5));

        /*let g = f!(x, y, z, 1.);
        println!("Int 3 = {}", rn_integral(&g, vec![fset![f!(y, z, 0.), f!(y, z, 1.-z)].wrap(), set![0, 2].wrap(), set![0, 1].wrap()], 2000));

        println!("Int 3,2 = {}", rn_integral(&g, vec![set![0, 5].wrap(), fset![f!(x, 0.), f!(x, 5.-x)].wrap(), fset![f!(x, y, 0.), f!(x, y, 5.-x-y)].wrap()], 100_000));*/
    }
    #[test]
    fn surfaces() {
        setup!();
        let v = vector_function!(x, y, z, x, y, z);
        let s:Surface = surface!(u, v, u.sin()*v.cos(), u.sin()*v.sin(), u.cos(), 0, PI/2., 0, 2.*PI);
        let rho = surface!(u, v, u+v, u*v, u.powf(v), set![0, 2.*PI], set![0, 10]);
        assert!(near!(s(PI, PI/2.), vector!(0, 0, -1)));

        let b:f64 = surface_integral!(v, s, Simpson(0.05));
        println!("a = {}", s.area());
        println!("b = {}", b);
    }

    #[test]
    fn integral_test() {
        setup!();
        let f = f!(x, x.powi(2));
        let g = f!(x, y, x*y);
        let a = integral!(f, set![0, 2], Simpson13(10));
        let b = integral!(g, set![0, 2], set![0, 3]);

        println!("a = {}    b = {}", a, b);
    }
}
