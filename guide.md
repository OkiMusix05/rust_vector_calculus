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

When you create a scalar function, the expression is automatically saved as a string, which you can access as `f.expression()` , and functions can be cloned with `f.clone()`. To evaluate a function, run `f(1.,2.)`, for example, which means these evaluate like regular Rust functions — it’s worth noting that aside from that, you can also pass a Vector as argument to a scalar function and it will evaluate perfectly as long as the dimensions match.

### Derivatives

You can evaluate a function’s partial derivatives too with the respective macro, like `ddx!`  for $\frac{\partial}{\partial x}$, and similarly for $y$, except for $\frac{\partial}{\partial z}$ in an $\mathbb{R}^2$ function, which will panic.

```rust
let a:f64 = ddy!(g, 2, 3.5, PI);
```

Note: These partial derivative macros return numbers, not functions.

### Integrals

For integrating functions, there’s the `integral!` macro, which can be called in many ways depending on the context, but all are equally valid. Here’s the general structure

```rust
integral!(f1:Function, bounds:Sets); // 1D function
integral!(f1:Function, a:f64, b:f64); // 1D function
integral!(f1:Function, bounds:Sets, method:IntegrationMethod); // 1D function
integral!(f2:Function, x_bounds:Sets, y_bounds:Sets) // 2D function
integral!(f2:Function, x_bounds:Sets, y_bounds:Sets, method:MultipleIntegrationMethod) // 2D function
integral!(f3:Function, x_bounds:Sets, y_bounds:Sets, z_bounds:Sets); // 3D function
integral!(f3:Function, x_bounds:Sets, y_bounds:Sets, z_bounds:Sets, method:MultipleIntegrationMethod); // 3D function
```

where `Sets` means it can be either a `Set` or an `FSet` , and where `IntegrationMethod` and `MultipleIntegrationMethod` are enums. For single variable integration methods you can look up the chart on Line Integrals, and for multiple there are the following methods:

| Method | Description |
| --- | --- |
| Monte Carlo | Random method, default is 400 points |
| Mid Point | Classic method, requires a $\delta$, recommended is 0.05 |
| Simpson | Better than Mid Point, also requires a $\delta$ |

Note: Integration for 3D functions with non-constant limits is not yet implemented.

Note: The `setup!()` macro was added to import these enum variants so you don’t have to write `IntegrationMethod::MonteCarlo(n)` every time, for example, and can just write `MonteCarlo(m)`.

### Gradient

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

### Limits

When taking the limit of 2D or 3D scalar functions, the value is approximated by finding the average of the function at $f(x_0\pm\Delta,y_0\pm\Delta)$. For this, you can use the `limit!` macro as so

```rust
let f:Function = f!(x, y, 2.*x*y);
let _:f64 = limit(f => 2, 3)
```

which uses specific notation (`=> ,`) to represent $\lim_{(x,y)\to(2,3)}f$ in this case.

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

These functions also encode their component scalar functions as strings, and you can access them as `F.expression_f1` and similarily for $f_2$ and $f_3$; they also evaluate like regular Rust functions, as well as evaluating on Vectors, like `F(v)`  if `v:Vector`.

Vector Functions also have methods to take their partial derivatives, although they’re named slightly differently, as the macros `ddxv!` where you can see the *v* added to indicate it is the partial derivative of a vector function. The more important methods, however, are *curl* and *div*, which also have their respective macros representing the operations $\nabla\times F$ and $\nabla\cdot F$.

```rust
let b:f64 = ddyv!(F, 1, 2);
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

There are also `FSet`’s, which are sets that instead of scalars contain two functions, and these are usually used when integrating over non-constant bounds. You can create one like so:

```rust
let y_bounds:FSet = fset![f!(x, 0.), f!(x, x.powi(2))];
```

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
let sigma:Contour = contour!(curve, space);
```

Contours, like parametric curves, also have a derivative `ddt` method, and the same macro applies here in the same way. Furthermore, like sets, contours have a `c.linspace(n)` method which returns a vector too.

Added to that, contours have a `c.bounds()` method that returns a tuple with two elements containing the start and end of the set.

Note: Contours are also evaluated like regular rust functions (`c(PI/2.)`) and return Vectors like parametric curves.

Note: Contours have a `c.len()` method which returns the length of the contour.

## Line Integrals

$$
\int_C F(x,y)\cdot ds
$$

Line integrals provide a way to integrate scalar or vector functions over the path of a contour. To do this, you can use the following macro

```rust
let _:f64 = line_integral!(f, c)
```

Where `f` can be either a `Function` or a `VectorFunction`, and `c` is a `Contour` of the apropiate dimention. However, it is possible to integrate a 3D vector field over a 2D contour.

By default, line integrals use the Gauss-Legendre Quadrature, but if you’d like another integration method you can do so as such:

```rust
 let _:f64 = line_integral!(f, c, IntegrationMethod::Simpson13);
```

where `IntegrationMethod` is an enum that provides the following methods

| Method | Description |
| --- | --- |
| Riemann | The classic one — you need to provide $n$: the number of partitions |
| Simpson 1/3 | The more accurate one — also needs an $n$ (has to be even) |
| Gauss-Legendre | The faster one, done with 5 points |

Note: If the vector function came from using the `grad!` macro, the line integral will automatically be computed using its potential function, as per the fundamental theorem of line integrals

$$
\oint_\sigma\nabla f\cdot ds=f(\sigma(t_1))-f(\sigma(t_0))
$$

If, however, the vector function comes from a gradient but it wasn’t obtained through the `grad!` macro (done by checking its curl), and $\sigma(t_1)=\sigma(t_0)$, by this same theorem the line integral will return 0.

$$
\oint_\sigma\nabla f\cdot ds = 0
$$

## Surfaces

Parametric surfaces are functions $S:\mathbb{R}^2\to\mathbb{R}^3$, and you can create one like this:

```rust
let s:ParametricSurface = parametric_surface!(u, v, u.sin()*v.cos(), u.sin()*v.sin(), u.cos());
```

However, parametric surfaces aren’t surfaces yet, because you have to limit its variables. If you already have a parametric surface you can create a surface with one and two sets like this:

```rust
let S:Surface = surface!(s, set![0, PI], fset![f!(u,v, 0.), f!(u, v, u+v)]);
```

If not, and all the bounds are constant, you can use the same macro and create one from scratch:

```rust
let S:Surface = surface!(u, v, u.sin()*v.cos(), u.sin()*v.sin(), u.cos(), 0, PI/2., 0, 2.*PI);
```

Finally, if you don’t have a parametric surface already and your bounds are not constant, you can use the macro as such:

```rust
let S:Surface = surface!(u, v, u+v, u*v, u.powf(v), fset![...], set![...]);
```

Surfaces have an `area` method, which calculates the area of a surface with the montecarlo method using 400 points. You can call it like `S.area()`.

## Surface Integrals

Surface integrals can integrate either scalar or vector functions. For this, there’s the `surface_integral!` macro, that you can use like this:

```rust
let _:f64 = surface_integral!(f, s); // f:Function 3D
let _:f64 = surface_integral!(v, s, 100); //v:VectorFunction 3D
```

where `s` is a `Surface` . The default number of points for the integral is 200, and it uses the Monte Carlo method, although you can specify another $n$ yourself too.

## Helpers
The `near!(a,b;e)` macro can be applied to vectors or numbers to determine if they're close enough to each other with an error e. You can also run it like `near!(a,b)`, which whill use an error of $1\times10^{-5}$ by default.