# Guide
This document is designed to teach all the functions on the crate.
## Vectors

Vectors here represent a vector in either $\mathbb{R}^2$ or $\mathbb{R}^3$, depending which object you are using. Vectors can be obtained as the result from a vector function, a parametric curve, an operation, etc.

To create a vector, you can use the `vector!` macro. The type of vector that comes from this depends on if you input 2 or 3 numbers. However, the type for both of this is `Vector`.

```rust
let u:Vector = vector!(2, 3);
let v:Vector = vector!(3.5, PI, 2);
```

If you want to access the $x$ and $y$ coordinates, you can do it as `v.x()` or `v.y()` , although the $z$ coordinate in 2D vectors is always 0.

There’s also operations you can do, for example the dot product between two vectors, expressed as `u*v` .

Along side scalar product, there’s also the cross product written as `u%v` , which always returns a Vector in $\mathbb{R}^3$.

To take the modulus of a vector, there are 3 ways:

1. `modulus(&v)` the function by reference
2. `md!(v)` the macro
3. `!v` , the not operator `!`, which when applied to vectors it returns the modulus.

## Scalar Functions

You can also create scalar functions $f(x,y)$ or $f(x,y,z)$. For this, use the `f!` macro, which depending on the arguments you pass, the type of the function: $\mathbb{R}^2$ or $\mathbb{R}^3$.

```rust
let f:Function = f!(x, y, x*y);
let g:Function = f!(x, y, z, x.powi(2)*y + z);
let _:f64 = g(1., 2., 2.);
```

When you create a scalar function, the expression is automatically saved as a string, which you can access as `f.expression()` , and functions can be cloned with `f.clone()`. To evaluate a function, run `f(1.,2.)`, for example, which means these evaluate like regular Rust functions.

You can evaluate its partial derivatives too with the respective macro, like `ddx!`  for $\frac{\partial}{\partial x}$, and similarly for $y$, except for $z$ in an $\mathbb{R}^2$ function, which will panic.

```rust
let a:f64 = ddy!(g, 2, 3.5, PI);
```

Note: These partial derivative macros return numbers, not functions.

From scalar functions you can also create *vector functions*, although in these cases the string expressions are not updated to be the correct partial derivatives. There is a specific macro for this

```rust
let del_g:VectorFunction = grad!(g);
```

$$
\nabla g = \begin{bmatrix}
\frac{\partial}{\partial x}\\
\frac{\partial}{\partial y}
\end{bmatrix}g
$$

Note that creating a vector function like this does not take ownership of the original function.

Note: Contrary to the partial derivative macros, the gradient macro does return a function.

## Vector Functions

$$
F(x,y)=\langle f_1(x,y),f_2(x,y)\rangle
$$

Vector Functions can be either in $\mathbb{R}^2$ or $\mathbb{R}^3$; they take in a $n$ number of arguments and return a vector.

You can create a Vector Function using the `vector_function!` macro like this:

```rust
let F:VectorFunction = vector_function!(x, y, x.powi(2), x*y);
let G:VectorFunction = vector_function!(x, y, z, 2.*z, x.sqrt(), x + y);
```

where the first parameters are the variables to use, and the last ones are the expressions for its component functions.

These functions also encode their component scalar functions as strings, and you can access them as `F.expression_f1` and similarily for $f_2$ and $f_3$; they also evaluate like regular Rust functions.

Vector Functions also have methods to take their partial derivatives, although they’re named slightly differently, as the macros `dvdy!` where you can see the *v* added to indicate it is the partial derivative of a vector function. The more important methods, however, are *curl* and *div*, which also have their respective macros representing the operations $\nabla\times F$ and $\nabla\cdot F$.

```rust
let b:f64 = dvdy!(F, 1, 2);
let _:Vector = curl!(F, 0, PI);
let c:f64 = !curl!(G, 1, PI, 2);
```

Note that in the last line of the code shown, the `!` operator is used on the vector produced by the curl macro to obtain the magnitude (modulus) of the rotational of the vector function $G$ at the point $(1,\pi,2)$.
## Contours

Contours consists of two parts: Parametric Curves and Sets.

### Parametric Curves

Parametric curves are a function that goes from $\mathbb{R}\to\mathbb{R}^n$, where here $n=2,3$. You can create a parametric curve using the`curve!` macro:

```rust
let curve:ParametricCurve = curve!(t, t.cos(), t.sin()); 
```

Parametric curves have a derivative method, which you can call like `curve.ddt(1.5)` or using the macro `ddt!(curve, t)`, for example, aside from also saving their expressions as strings like *Vector Functions*.

You can evaluate them in a point as any function, in the form of `curve(PI)`, and it returns a `Vector`.

### Sets

$$
t\in[0, 2\pi]
$$

A set in this library consists of the space a variable belongs to. Sets are used in conjunction with parametric curves to provide the bounds for the independent variable of the curve. You can create a set like this:

```rust
let t_space:Set = set![0, PI];
```

where the two numbers indicate the continous one-dimensional space where a variable can move.

You can access the start and the end of this set through `t_space.i` and `t_space.f`.

Sets also have a linspace method that you can call like `set.linspace(n)` where $n$ is the number of steps it takes when creating a `Vec` with its values in between.

### Contours

When using these parametric curves for a *line integral*, for example, the preferred method is to use a contour, which you can create like this

```rust
let c:Contour = contour!(t, t, t.powi(2), 0, 1); // in R^2
let rho:Contour = contour!(t, t.cos(), t.sin(), t, 0, PI); // in R^3
```

where the first parameters belong to the parametric curve and the last two to the set.

However, if you already have made two variables for the curve and the set you can make a contour using the same macro like this:

```rust
let curve:ParametricCurve = curve!(...);
let space:Set = set![...];
let sigma:Contour = contour!(curve, spaec);
```

Contours, like parametric curves, also have a derivative `ddt` method, and the same macro applies here in the same way. Furthermore, like sets, contours have a `c.linspace(n)` method which returns a vector too.

Added to that, contours have a `c.bounds()` method that returns a tuple with two elements containing the start and end of the set.

Note: Contours are also evaluated like regular rust functions (`c(PI/2.)`) and return Vectors like parametric curves.