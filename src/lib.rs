#![feature(unboxed_closures, fn_traits)]
use std::fmt::{Display, Formatter};
use dyn_clone::DynClone;

const Δ:f64 = 5e-6;

// ----- VECTORS -----
#[derive(Copy, Clone, Debug)]
pub struct Vector2 {
    pub x:f64,
    pub y:f64
}
#[derive(Copy, Clone, Debug)]
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
#[derive(Copy, Clone, Debug)]
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
pub struct Function1D {
    pub f:Box<dyn F1DClone<Output=f64>>,
    pub expression:String
}
impl Function1D {
    fn call(&self, x:f64) -> f64 {
        (self.f)(x)
    }
}
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
impl Clone for Function1D {
    fn clone(&self) -> Function1D {
        Function1D {
            f: dyn_clone::clone_box(&*self.f),
            expression: String::from(&*self.expression)
        }
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
    OneD(Function1D),
    TwoD(Function2D),
    ThreeD(Function3D)
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
        Function::OneD(Function1D {
            f: Box::new(|$x:f64| $f),
            expression: String::from(stringify!($f))
        })
    };
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
pub struct VectorFunction2D {
    pub f1:Box<dyn Fn(f64, f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64, f64) -> f64>,
    pub expression_f2:String,
    pub potential: Option<Function>,
    pub potential_expression: String
}
pub struct VectorFunction3D {
    pub f1:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f2:String,
    pub f3:Box<dyn Fn(f64, f64, f64) -> f64>,
    pub expression_f3:String,
    pub potential: Option<Function>,
    pub potential_expression: String
}
pub enum VectorFunction {
    TwoD(VectorFunction2D),
    ThreeD(VectorFunction3D)
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
    pub fn expression(&self) -> String {
        match self {
            VectorFunction::TwoD(v) => v.clone().potential_expression.clone(),
            VectorFunction::ThreeD(v) => v.clone().potential_expression.clone()
        }
    }
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
            expression_f2: String::from(stringify!($f2)),
            potential: Option::None,
            potential_expression: String::from("")
        })
    };
    ($x:ident, $y:ident, $z:ident, $f1:expr, $f2:expr, $f3:expr) => {
        VectorFunction::ThreeD(VectorFunction3D {
            f1: Box::new(|$x:f64, $y:f64, $z:f64| $f1),
            expression_f1: String::from(stringify!($f1)),
            f2: Box::new(|$x:f64, $y:f64, $z:f64| $f2),
            expression_f2: String::from(stringify!($f2)),
            f3: Box::new(|$x:f64, $y:f64, $z:f64| $f3),
            expression_f3: String::from(stringify!($f3)),
            potential: Option::None,
            potential_expression: String::from("")
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
impl FnOnce<(Vector,)> for VectorFunction {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (Vector,)) -> Self::Output {
        match self {
            VectorFunction::TwoD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::TwoD(Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
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
                    Vector::TwoD(v) => Vector::TwoD(Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
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
                    Vector::TwoD(v) => Vector::TwoD(Vector2::new((f.f1)(v.x, v.y), (f.f2)(v.x, v.y))),
                    Vector::ThreeD(_) => panic!("3D vector passed to 2D vector function")
                }
            }
            VectorFunction::ThreeD(f) => {
                match args.0 {
                    Vector::TwoD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, 0.), (f.f2)(v.x, v.y, 0.), (f.f3)(v.x, v.y, 0.))),
                    Vector::ThreeD(v) => Vector::ThreeD(Vector3::new((f.f1)(v.x, v.y, v.z), (f.f2)(v.x, v.y, v.z), (f.f3)(v.x, v.y, v.z)))
                }
            }
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
        Function::OneD(_) => panic!("Gradient vector not defined for 1D function"),
        Function::TwoD(_) => {
            let f1 = f.clone();
            let f2 = f.clone();
            VectorFunction::TwoD(VectorFunction2D {
                expression_f1: format!("ddx({})", f.expression()),
                expression_f2: format!("ddy({})", f.expression()),
                potential_expression: f.expression(),
                f1: Box::new(move |x:f64, y:f64| ddx_s(&f1, vec![x, y])),
                f2: Box::new(move |x:f64, y:f64| ddy_s(&f2, vec![x, y])),
                potential: Some(f.clone()),
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
                potential_expression: f.expression(),
                f1: Box::new(move |x:f64, y:f64, z:f64| ddx_s(&f1, vec![x, y, z])),
                f2: Box::new(move |x:f64, y:f64, z:f64| ddy_s(&f2, vec![x, y, z])),
                f3: Box::new(move |x:f64, y:f64, z:f64| ddz_s(&f3, vec![x, y, z])),
                potential: Some(f.clone()),
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
pub struct ParametricCurve2D {
    pub f1:Box<dyn Fn(f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64) -> f64>,
    pub expression_f2:String,
}
impl ParametricCurve2D {
    pub fn ddt(&self, t:f64) -> Vector {
        Vector::TwoD(Vector2::new(((self.f1)(t + Δ) - (self.f1)(t))/Δ, ((self.f2)(t + Δ) - (self.f2)(t))/Δ))
    }
}
impl FnOnce<(f64,)> for ParametricCurve2D {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        Vector::TwoD(Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
impl FnMut<(f64,)> for ParametricCurve2D {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        Vector::TwoD(Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
impl Fn<(f64,)> for ParametricCurve2D {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        Vector::TwoD(Vector2::new((self.f1)(args.0), (self.f2)(args.0)))
    }
}
pub struct ParametricCurve3D {
    pub f1:Box<dyn Fn(f64) -> f64>,
    pub expression_f1:String,
    pub f2:Box<dyn Fn(f64) -> f64>,
    pub expression_f2:String,
    pub f3:Box<dyn Fn(f64) -> f64>,
    pub expression_f3:String
}
impl ParametricCurve3D {
    pub fn ddt(&self, t:f64) -> Vector {
        Vector::ThreeD(Vector3::new(((self.f1)(t + Δ) - (self.f1)(t))/Δ, ((self.f2)(t + Δ) - (self.f2)(t))/Δ, ((self.f3)(t + Δ) - (self.f3)(t))/Δ))
    }
}
impl FnOnce<(f64,)> for ParametricCurve3D {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
impl FnMut<(f64,)> for ParametricCurve3D {
    extern "rust-call" fn call_mut(&mut self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
impl Fn<(f64,)> for ParametricCurve3D {
    extern "rust-call" fn call(&self, args: (f64,)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0), (self.f2)(args.0), (self.f3)(args.0)))
    }
}
pub enum ParametricCurve {
    TwoD(ParametricCurve2D),
    ThreeD(ParametricCurve3D)
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
                write!(f, "⟨{}, {}⟩", sigma.expression_f1, sigma.expression_f2)
            }
            ParametricCurve::ThreeD(sigma) => {
                write!(f, "⟨{}, {}, {}⟩", sigma.expression_f1, sigma.expression_f2, sigma.expression_f3)
            }
        }
    }
}
#[macro_export]
macro_rules! curve {
    ($t:ident, $f1:expr, $f2:expr) => {
        ParametricCurve::TwoD(ParametricCurve2D {
            f1: Box::new(|$t:f64| $f1),
            expression_f1: String::from(stringify!($f1)),
            f2: Box::new(|$t:f64| $f2),
            expression_f2: String::from(stringify!($f2)),
        })
    };
    ($t:ident, $f1:expr, $f2:expr, $f3:expr) => {
        ParametricCurve::ThreeD(ParametricCurve3D {
            f1: Box::new(|$t:f64| $f1),
            expression_f1: String::from(stringify!($f1)),
            f2: Box::new(|$t:f64| $f2),
            expression_f2: String::from(stringify!($f2)),
            f3: Box::new(|$t:f64| $f3),
            expression_f3: String::from(stringify!($f3))
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
#[macro_export]
macro_rules! set {
    ($i:expr, $f:expr) => {Set {
        i: $i as f64,
        f: $f as f64
    }};
}
impl Display for Set {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.5}, {:.5}]", self.i, self.f)
    }
}

// ----- CONTOURS -----
pub struct Contour2D {
    pub f_t: ParametricCurve2D,
    pub lim: Set
}
pub struct Contour3D {
    pub f_t: ParametricCurve3D,
    pub lim: Set
}
pub enum Contour {
    TwoD(Contour2D),
    ThreeD(Contour3D)
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
}
impl Display for Contour {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Contour::TwoD(c) => {
                write!(f, "⟨{}, {}⟩", c.f_t.expression_f1, c.f_t.expression_f2)
            }
            Contour::ThreeD(c) => {
                write!(f, "⟨{}, {}, {}⟩", c.f_t.expression_f1, c.f_t.expression_f2, c.f_t.expression_f3)
            }
        }
    }
}
#[macro_export]
macro_rules! contour {
    ($t:ident, $f1:expr, $f2:expr, $t0:expr, $t1:expr) => {
        Contour::TwoD(Contour2D {
            f_t: ParametricCurve2D {
                f1: Box::new(|$t:f64| { $f1 }),
                expression_f1: String::from(stringify!($f1)),
                f2: Box::new(|$t:f64| { $f2 }),
                expression_f2: String::from(stringify!($f2))
            },
            lim: set![$t0, $t1]
        })
    };
    ($t:ident, $f1:expr, $f2:expr, $f3:expr, $t0:expr, $t1:expr) => {
        Contour::ThreeD(Contour3D {
            f_t: ParametricCurve3D {
                f1: Box::new(|$t:f64| { $f1 }),
                expression_f1: String::from(stringify!($f1)),
                f2: Box::new(|$t:f64| { $f2 }),
                expression_f2: String::from(stringify!($f2)),
                f3: Box::new(|$t:f64| { $f3 }),
                expression_f3: String::from(stringify!($f3))
            },
            lim: set![$t0, $t1]
        })
    };
    ($curve:expr, $set:expr) => {
        match $curve {
            ParametricCurve::TwoD(c) => {
                Contour::TwoD(Contour2D {
                    f_t: c,
                    lim: $set
                })
            },
            ParametricCurve::ThreeD(c) => {
                Contour::ThreeD(Contour3D {
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
#[macro_export]
macro_rules! near {
    ($a:expr, $b:expr) => {
        $a > $b - 2.*Δ && $a < $b + 2.*Δ
    };
    ($a:expr, $b:expr; $e:expr) => {
        $a > $b - $e && $a < $b + $e
    };
}
#[macro_export]
macro_rules! near_v {
    ($u:expr, $v:expr) => {
        match ($u, $v) {
            (Vector::TwoD(u), Vector::TwoD(v)) => {
                near!(u.x, v.x) && near!(u.y, v.y)
            },
            (Vector::ThreeD(u), Vector::ThreeD(v)) => {
                near!(u.x, v.x) && near!(u.y, v.y) && near!(u.z, v.z)
            },
            (Vector::TwoD(u), Vector::ThreeD(v)) => {
                near!(u.x, v.x) && near!(u.y, v.y) && near!(0.0, v.z)
            },
            (Vector::ThreeD(u), Vector::TwoD(v)) => {
                near!(u.x, v.x) && near!(u.y, v.y) && near!(u.z, 0.0)
            }
        }
    };
    ($u:expr, $v:expr; $e:expr) => {
        match ($u, $v) {
            (Vector::TwoD(u), Vector::TwoD(v)) => {
                near!(u.x, v.x; $e) && near!(u.y, v.y; $e)
            },
            (Vector::ThreeD(u), Vector::ThreeD(v)) => {
                near!(u.x, v.x; $e) && near!(u.y, v.y; $e) && near!(u.z, v.z; $e)
            },
            (Vector::TwoD(u), Vector::ThreeD(v)) => {
                near!(u.x, v.x; $e) && near!(u.y, v.y; $e) && near!(0.0, v.z; $e)
            },
            (Vector::ThreeD(u), Vector::TwoD(v)) => {
                near!(u.x, v.x; $e) && near!(u.y, v.y; $e) && near!(u.z, 0.0; $e)
            },
        }
    };
}
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
// General Function Wrapper
enum _G<'s> {
    Function(&'s Function),
    VectorFunction(&'s VectorFunction)
}
trait Wrap {
    fn wrap(&self) -> _G;
} // Wrapper for General Function
impl Wrap for Function {
    fn wrap(&self) -> _G {
        _G::Function(self)
    }
}
impl Wrap for VectorFunction {
    fn wrap(&self) -> _G {
        _G::VectorFunction(&self)
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
pub fn line_integral(g:_G, c:&Contour, method:IntegrationMethod) -> f64 {
    let (t0, t1) = c.bounds();
    let mut ft:Box<dyn Fn(f64)->f64> = Box::new(|_:f64| f64::NAN);
    match g {
        _G::Function(f) => {
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
        _G::VectorFunction(v) => {
            let (c1,c0) = (c(t1), c(t0));
            let (half, third) = ((71./67.)*((t1-t0)/2.), (129./131.)*((t1-t0)/3.));
            match v {
                VectorFunction::TwoD(vf) => {
                    match c {
                        Contour::TwoD(_) => {
                            if let Some(f) = vf.potential.clone() {
                                if near_v!(c1, c0; Δ*1e-7) {
                                    return 0.
                                } else {
                                    return f(c1) - f(c0);
                                }
                            } else {
                                let (half, third) = (!curl!(v, c(half).x(), c(half).y()), !curl!(v, c(third).x(), c(third).y()));
                                if near!(half, 0.0) && near!(third, 0.0) && near_v!(c1, c0){
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
                        if near_v!(c1, c0; Δ*1e-7) {
                            return 0.
                        } else {
                            return f(c1) - f(c0);
                        }
                    }
                    match c {
                        Contour::TwoD(_) => {
                            let (half, third) = (!curl!(v, c(half).x(), c(half).y(), 0.), !curl!(v, c(third).x(), c(third).y(), 0.));
                            if near!(half, 0.0) && near!(third, 0.0) && near_v!(c1, c0){
                                return 0.
                            } else {
                                ft = Box::new(move |t:f64| {
                                    return v(c(t))*ddt!(c, t)
                                });
                            }
                        }
                        Contour::ThreeD(_) => {
                            let (half, third) = (!curl!(v, c(half).x(), c(half).y(), c(half).z()), !curl!(v, c(third).x(), c(third).y(), c(third).z()));
                            if near!(half, 0.0) && near!(third, 0.0) && near_v!(c1, c0){
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
macro_rules! integral {
    ($f:expr, $s:expr) => {
        integral_1d(&$f, &$s, IntegrationMethod::GaussLegendre)
    };
    ($f:expr, $a:expr, $b:expr) => {
        integral_1d(&$f, &set![$a, $b], IntegrationMethod::GaussLegendre)
    };
    ($f:expr, $a:expr, $b:expr, $m:expr) => {
        integral_1d(&$f, &set![$a, $b], $m)
    };
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

// ----- DOUBLE INTEGRAL -----


// ----- SURFACES -----
pub struct ParametricSurface { // All supposed to be Function2D
    f1:Function,
    expression_f1:String,
    f2:Function,
    expression_f2:String,
    f3:Function,
    expression_f3:String
}
pub struct Surface {
    f:ParametricSurface,
    u_lim:Set,
    v_lim:Set
}
impl Surface {
    pub fn ddu(&self, u:f64, v:f64) -> Vector {
        Vector::ThreeD(Vector3::new(((self.f.f1)(u + Δ, v) - (self.f.f1)(u, v))/Δ, ((self.f.f2)(u + Δ, v) - (self.f.f2)(u, v))/Δ, ((self.f.f3)(u + Δ, v) - (self.f.f3)(u, v))/Δ))
    }
    pub fn ddv(&self, u:f64, v:f64) -> Vector {
        Vector::ThreeD(Vector3::new(((self.f.f1)(u, v + Δ) - (self.f.f1)(u, v))/Δ, ((self.f.f2)(u, v + Δ) - (self.f.f2)(u, v))/Δ, ((self.f.f3)(u, v + Δ) - (self.f.f3)(u, v))/Δ))
    }
}
impl FnOnce<(f64, f64)> for ParametricSurface {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl FnMut<(f64, f64)> for ParametricSurface {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl Fn<(f64, f64)> for ParametricSurface {
    extern "rust-call" fn call(&self, args: (f64, f64)) -> Self::Output {
        Vector::ThreeD(Vector3::new((self.f1)(args.0, args.1), (self.f2)(args.0, args.1), (self.f3)(args.0, args.1)))
    }
}
impl FnOnce<(f64, f64)> for Surface {
    type Output = Vector;

    extern "rust-call" fn call_once(self, args: (f64, f64)) -> Self::Output {
        (self.f)(args.0, args.1)
    }
}
impl FnMut<(f64, f64)> for Surface {
    extern "rust-call" fn call_mut(&mut self, args: (f64, f64)) -> Self::Output {
        (self.f)(args.0, args.1)
    }
}
impl Fn<(f64, f64)> for Surface {
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
            u_lim: set![$ui, $uf],
            v_lim: set![$vi, $vf]
        }
    };
    ($p:expr, $ul:expr, $vl:expr) => {
        Surface {
            f: $p,
            u_lim: $ul,
            v_lim: $vl
        }
    }
}

// ----- SURFACE INTEGRAL -----
/*pub fn surface_integral(f:&Function, s:&Surface) -> f64 {
    match f {
        Function::ThreeD(_) => {
            let ft = Box::new(|t:f64| {
               !(s.ddu())
            });
        },
        _ => panic!("No surface integrals for 1D and 2D functions")
    }
    f64::NAN
}*/


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
        assert!(near!(integral!(h, 1., E), 1.));
        assert!(near!(ddx!(h, 2.), -1./4.));
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
        println!("∇g(1, 2, 3) = {}", del_g(1., 2., 3.));
        assert_eq!(del_g.potential(vec![1., 2., 3.]), g(1., 2., 3.));
    }

    #[test]
    fn contours() {
        let sigma:ParametricCurve = curve!(t, t.powi(2), 2.*t);
        let space:Set = set![0., 2.*PI];
        //println!("sigma = {}, space = {}", sigma, space);

        let c:Contour = contour!(t, t.cos(), t.sin(), 0, PI);
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
    fn surfaces() {
        let s:Surface = surface!(u, v, u.sin()*v.cos(), u.sin()*v.sin(), u.cos(), 0, PI/2., 0, 2.*PI);
        println!("s(PI, PI/2) = {}", s(PI, PI/2.));
    }
}
