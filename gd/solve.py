import sympy
import math
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

p0=1000000
x0=0.54
u0=0.61 #ml/s
n=108.5
t=320
r=8.314
l=0.24
di=0.0002
miu=0.000544
pi1=250000
j=0.0001
a=0.000003
pi0=100000
k=1
y0=0.46
aerfa=32
xl,ul,yl,piba=sympy.symbols("xl ul yl piba")
f1=u0*x0-ul*xl-a*j*(p0*(x0+xl)/2-piba*(y0+yl)/2)
print('f1:',sympy.simplify(f1))
f2=u0-ul-u0*x0+ul*xl-a*j*(p0*(1-(x0+xl)/2)-piba*(1-(y0+yl)/2))/aerfa
print('f2:',sympy.simplify(f2))
f3=(256*r*t*k*(u0-ul)/3*miu*l/(n*(math.pi)*di**4)+pi0**2)**0.5-piba
print('f3:',sympy.simplify(f3))
f4=(p0*xl-piba*yl)/((p0*xl-piba*yl)+(p0*(1-xl)-piba*(1-yl))/aerfa)-yl
print('f4:',sympy.simplify(f4))
# answer=sympy.solve([f1,f2,f3,f4],[xl,ul,yl,piba])
answer=sympy.solve([f1,f2,f3,f4],[yl])
print('answer:',answer)

