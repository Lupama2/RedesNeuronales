#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
File: ej_03.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

"""
Aproximando a la neurona como un capacitor, tenemos que
$$
C_m \frac{dV}{dt} = \frac{dQ}{dt}.
$$

Luego, una variación en la carga resulta en
$$
\Delta Q = C_m \Delta V.
$$

Podemos expresar a la capacitancia de la célula como la capacitancia por unidad de área multiplicada por el área de la neurona
$$
\begin{align}
C_m &= c A\\
 &= c 4 \pi R^2.
\end{align}
$$

Además, tenemos que la cantidad de carga se relaciona con el número de iones como
$$
N = \frac{Q}{F}.
$$

Entonces
$$
\begin{align}
\Delta N &= \frac{\Delta Q}{F}\\
 &= c 4 \pi R^2 \frac{\Delta V}{F}.
\end{align}
$$

Por último, calculamos el cambio en la concentración, siendo $\mathcal{V}$ el volumen de la célula
$$
\begin{align}
\Delta n &= \frac{\Delta N}{\mathcal{V}}\\
 &= \frac{c 4 \pi R^2 \Delta V}{ \frac{4}{3} \pi R^3 F}\\
 &= \frac{3c \Delta V}{RF}.
\end{align}
$$

Ahora para las unidades, tenemos que
$$
1 \text{ L} = 1 \text{ dm}^3,
$$
por lo que trabajaremos en dm para obtener una concentración en unidades de molaridad (M).

Convirtiendo unidades: $R = 15 \times 10^{-5}$ dm, $c = 1 \times 10^{-4}$ F dm$^{-2}$ y $\Delta V = 1 \times 10^{-1}$ V resulta
$$
\begin{align}
\Delta n &= \frac{3 \times 10^{-5}}{15 \times 105 \times 10^{-5}} \text{ M}\\
 &= 1.9 \text{ mM}.
\end{align}
$$

Recordemos que las concentraciones del Ej. 1 son de órdenes entre de magnitud entre $10^{1}-10^{2}$ mM. Por lo que este cambio en concentración es pequeño en comparación, a pesar de que conlleva un cambio de tensión de 100 mV, siendo los potenciales de reversión del órden de $10^{2}$ mV. Esto es un cambio apreciable de tensión.

"""