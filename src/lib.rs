#![feature(unboxed_closures, fn_traits)]
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
const Δ:f64 = 5e-6;

// ----- VECTORS -----
#[derive(Copy, Clone, Debug)]
pub struct _Vector2 {
    pub x:f64,
    pub y:f64
}
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
pub enum Vector {
    TwoD(_Vector2),
    ThreeD(_Vector3)
}
impl Vector {
    pub fn new_2d(x:f64, y:f64) -> Self {
        Vector::TwoD(_Vector2 { x, y, })
    }
    pub fn new_3d(x:f64, y:f64, z:f64) -> Self {
        Vector::ThreeD(_Vector3 { x, y, z })
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
impl PartialEq for Vector {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Vector::TwoD(u), Vector::TwoD(v)) => if u.x == v.x && u.y == v.y { true } else {false},
            (Vector::ThreeD(u), Vector::ThreeD(v)) => if u.x == v.x && u.y == v.y && u.z == v.z { true } else { false },
            (_, _) => false
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
// Cross product for Vector
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
pub trait F1DClone: DynClone + Fn(f64,) -> f64 {
    fn call(&self, x: f64) -> f64;
}
pub trait F2DClone: DynClone + Fn(f64, f64) -> f64 {
    fn call(&self, x: f64, y: f64) -> f64;
}
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
pub struct _Function1D {
    pub f:Box<dyn F1DClone<Output=f64>>,
    pub expression:String
}
impl _Function1D {
    fn call(&self, x:f64) -> f64 {
        (self.f)(x)
    }
}
pub struct _Function2D {
    pub f:Box<dyn F2DClone<Output=f64>>,
    pub expression:String
}
impl _Function2D {
    fn call(&self, x:f64, y:f64) -> f64 {
        (self.f)(x, y)
    }
}
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
pub enum Function {
    OneD(_Function1D),
    TwoD(_Function2D),
    ThreeD(_Function3D)
}
impl Function {
    pub fn expression(&self) -> String {
        match self {
            Function::OneD(f) => f.clone().expression,
            Function::TwoD(f) => f.clone().expression,
            Function::ThreeD(f) => f.clone().expression
        }
    }
}
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
pub fn ddz_s(f:&Function, args:Vec<f64>) -> f64 {
    match f {
        Function::OneD(f) => {
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
#[macro_export]
macro_rules! ddx {
    ($f:expr, $x:expr) => {ddx_s(&$f, vec![$x as f64])};
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
            Function::OneD(f) => Function::OneD(f.clone()),
            Function::TwoD(f) => Function::TwoD(f.clone()),
            Function::ThreeD(f) => Function::ThreeD(f.clone())
        }
    }
}
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
#[macro_export]
macro_rules! limit {
    ($f:expr => $x:expr,$y:expr) => {limit_s(&$f, vec![$x as f64, $y as f64])};
    ($f:expr => $x:expr,$y:expr, $z:expr) => {limit_s(&$f, vec![$x as f64, $y as f64, $z as f64])};
}

// ----- VECTOR FUNCTIONS -----
#[derive(Clone)]
pub struct _VectorFunction2D {
    pub f1:Function,
    pub f2:Function,
    pub potential: Option<Function>,
}
#[derive(Clone)]
pub struct _VectorFunction3D {
    pub f1:Function,
    pub f2:Function,
    pub f3:Function,
    pub potential: Option<Function>,
}
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
pub fn ddz_v(v:&VectorFunction, args:Vec<f64>) -> Vector {
    match v {
        VectorFunction::TwoD(_) => {
            panic!("Can't take partial with respect to z of a 2D vector function")
        },
        VectorFunction::ThreeD(v) => {
            Vector::TwoD(_Vector2::new(((v.f1)(args[0], args[1], args[2] + Δ)-(v.f1)(args[0], args[1], args[2]))/Δ, ((v.f2)(args[0], args[1], args[2] + Δ)-(v.f2)(args[0], args[1], args[2]))/Δ))
        }
    }
}
#[macro_export]
macro_rules! ddxv {
    ($f:expr, $x:expr, $y:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddx_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddyv {
    ($f:expr, $x:expr, $y:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64])};
    ($f:expr, $x:expr, $y:expr, $z:expr) => {ddy_v(&$f, vec![$x as f64, $y as f64, $z as f64])};
}
#[macro_export]
macro_rules! ddzv {
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
        Function::OneD(_) => panic!("Gradient vector not defined for 1D function"),
        Function::TwoD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            VectorFunction::TwoD(_VectorFunction2D {
                f1: Function::TwoD(_Function2D {
                    f: Box::new(move |x:f64, y:f64| ddy_s(&f1, vec![x, y])),
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
#[macro_export]
macro_rules! grad {
    ($f:expr) => {
        grad(&$f)
    }
}

// ----- PARAMETRIC CURVES -----
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
macro_rules! ddt {
    ($f:expr, $t:expr) => {$f.ddt($t as f64)};
}

// ----- SETS -----
trait Super {
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
#[derive(Clone)]
pub struct FSet {
    pub i:Function,
    pub f:Function,
}
impl FSet {
    fn new(i:Function, f:Function) -> Self {
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
#[derive(Clone)]
enum _SuperSet<'s> {
    Set(&'s Set),
    FSet(&'s FSet)
}
#[macro_export]
macro_rules! set {
    ($i:expr, $f:expr) => {Set {
        i: $i as f64,
        f: $f as f64
    }};
}
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
#[derive(Clone)]
pub struct _Contour2D {
    pub f_t: _ParametricCurve2D,
    pub lim: Set
}
#[derive(Clone)]
pub struct _Contour3D {
    pub f_t: _ParametricCurve3D,
    pub lim: Set
}
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
pub fn type_of<T>(_: &T) -> &str {
    std::any::type_name::<T>().split("::").collect::<Vec<&str>>().last().unwrap()
}

pub enum __NV {
    Number(f64),
    Vector(Vector)
}
trait NVWrap {
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
enum __G<'s> {
    Function(&'s Function),
    VectorFunction(&'s VectorFunction)
}
trait GWrap {
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
#[macro_export]
macro_rules! line_integral {
    ($g:expr, $c:expr) => {
        line_integral($g.wrap(), &$c, IntegrationMethod::GaussLegendre)
    };
    ($g:expr, $c:expr, $m:expr) => {
        line_integral($g.wrap(), &$c, $m)
    };
}
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
pub enum __M {
    Single(IntegrationMethod),
    Multi(MultipleIntegrationMethod)
}
// Integration arguments wrapper
pub enum __W<'s> {
    Number(f64),
    SSet(_SuperSet<'s>),
    Method(__M),
}
trait WWrap {
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
    ($f:expr, $x:expr, $y:expr, $z:expr, $m:expr) => { // 3D - specified
        categorize_integrals(&$f, vec![$x.wwrap(), $y.wwrap(), $z.wwrap(), $m.wwrap()])
    };
}

// ----- SURFACES -----
#[derive(Clone)]
pub struct ParametricSurface { // All supposed to be Function2D
    f1:Function,
    expression_f1:String,
    f2:Function,
    expression_f2:String,
    f3:Function,
    expression_f3:String
}

#[derive(Clone)]
pub struct Surface<'s> {
    f:ParametricSurface,
    u_lim: _SuperSet<'s>,
    v_lim: _SuperSet<'s>
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
#[macro_export]
macro_rules! parametric_surface {
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr) => {
        ParametricSurface {
                f1: f!($u, $v, $f1),
                expression_f1: String::from(stringify!($f1)),
                f2: f!($u, $v, $f2),
                expression_f1: String::from(stringify!($f2)),
                f1: f!($u, $v, $f3),
                expression_f1: String::from(stringify!($f3)),
            }
    };
}
#[macro_export]
macro_rules! surface {
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr, $ui:expr, $uf:expr, $vi:expr, $vf:expr) => {
        Surface {
            f: ParametricSurface {
                f1: f!($u, $v, $f1),
                expression_f1: String::from(stringify!($f1)),
                f2: f!($u, $v, $f2),
                expression_f2: String::from(stringify!($f2)),
                f3: f!($u, $v, $f3),
                expression_f3: String::from(stringify!($f3)),
            },
            u_lim: set![$ui, $uf].wrap(),
            v_lim: set![$vi, $vf].wrap()
        }
    };
    ($u:ident, $v:ident, $f1:expr, $f2:expr, $f3:expr, $ul:expr, $vl:expr) => {
        Surface {
            f: ParametricSurface {
                f1: f!($u, $v, $f1),
                expression_f1: String::from(stringify!($f1)),
                f2: f!($u, $v, $f2),
                expression_f2: String::from(stringify!($f2)),
                f3: f!($u, $v, $f3),
                expression_f3: String::from(stringify!($f3)),
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
#[macro_export]
macro_rules! surface_integral {
    ($f:expr, $s:expr) => { surface_integral(&$f.wrap(), &$s, MultipleIntegrationMethod::MonteCarlo(400)) };
    ($f:expr, $s:expr, $m:expr) => { surface_integral(&$f.wrap(), &$s, $m) };
}

// ----- CONFIG -----
#[macro_export]
macro_rules! setup {
    () => {
        use $crate::{IntegrationMethod::{self, *}, MultipleIntegrationMethod::{self, *}};
        use std::f64::consts::{PI, E};
    };
}

// ----- HELPERS ------
#[macro_export]
macro_rules! sin {
    ($x:expr) => {$x.sin()};
}
#[macro_export]
macro_rules! cos {
    ($x:expr) => {$x.cos()};
}
#[macro_export]
macro_rules! tan {
    ($x:expr) => {$x.tan()};
}
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
