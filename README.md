Basic differentiable optical design primitives in python, built using JAX.

This allows the definition of an optical model, consisting of a sequence of
surfaces which interact with light in different ways. Rays can be traced
through the model, and the final ray position/directions can be differentiated
with respect to the parameters of the model, in order to efficiently optimise
the parameters

Compared to other optical design packages, this is much more primitive, doesn't
follow standard conventions of optical design, and doesn't do any of the normal
analysis you might want when designing an imaging lens.

This exists because I found other packages to be too restrictive, and have too
many assumptions which get in the way when designing optics for non-imaging
applications.

This is currently early-stage software, and I don't necessarily plan to polish
it beyond adding features needed for my own projects.

If you're interested anyway, have a look at the `notebooks` folder for typical
usage, and feel free to open issues if something doesn't work.

# license

see LICENSE.txt

```
jax_optics

Copyright (C) 2025  Thomas Nixon

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
