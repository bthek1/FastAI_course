# Foundations

Foundations

Author

Benedict Thekkel

``` python
import torch
import matplotlib.pyplot as plt
import random
```

### Callbacks as GUI events

``` python
import ipywidgets as widgets
```

From the [ipywidget docs](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html):

- *the button widget is used to handle mouse clicks. The on_click method of the Button can be used to register function to be called when the button is clicked*

``` python
w = widgets.Button(description='Click me')
```

``` python
w
```

``` python
def f(o): print('hi')
```

``` python
w.on_click(f)
```

*NB: When callbacks are used in this way they are often called “events”.*

### Creating your own callback

``` python
from time import sleep
from tqdm import trange
```

``` python
def slow_calculation():
    res = 0
    for i in trange(5):
        res += i*i
        sleep(1)
    return res
```

``` python
slow_calculation()
```

    100%|█████████████████████████| 5/5 [00:05<00:00,  1.00s/it]

    30

``` python
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        res += i*i
        sleep(1)
        if cb: cb(i)
    return res
```

``` python
def show_progress(epoch): print(f"Awesome! We've finished epoch {epoch}!")
```

``` python
slow_calculation(show_progress)
```

    Awesome! We've finished epoch 0!
    Awesome! We've finished epoch 1!
    Awesome! We've finished epoch 2!
    Awesome! We've finished epoch 3!
    Awesome! We've finished epoch 4!

    30

### Lambdas and partials

``` python
slow_calculation(lambda o: print(f"Awesome! We've finished epoch {o}!"))
```

    Awesome! We've finished epoch 0!
    Awesome! We've finished epoch 1!
    Awesome! We've finished epoch 2!
    Awesome! We've finished epoch 3!
    Awesome! We've finished epoch 4!

    30

``` python
def show_progress(exclamation, epoch): print(f"{exclamation}! We've finished epoch {epoch}!")
```

``` python
slow_calculation(lambda o: show_progress("OK I guess", o))
```

    OK I guess! We've finished epoch 0!
    OK I guess! We've finished epoch 1!
    OK I guess! We've finished epoch 2!
    OK I guess! We've finished epoch 3!
    OK I guess! We've finished epoch 4!

    30

``` python
def make_show_progress(exclamation):
    def _inner(epoch): print(f"{exclamation}! We've finished epoch {epoch}!")
    return _inner
```

``` python
slow_calculation(make_show_progress("Nice!"))
```

    Nice!! We've finished epoch 0!
    Nice!! We've finished epoch 1!
    Nice!! We've finished epoch 2!
    Nice!! We've finished epoch 3!
    Nice!! We've finished epoch 4!

    30

``` python
from functools import partial
```

``` python
slow_calculation(partial(show_progress, "OK I guess"))
```

    OK I guess! We've finished epoch 0!
    OK I guess! We've finished epoch 1!
    OK I guess! We've finished epoch 2!
    OK I guess! We've finished epoch 3!
    OK I guess! We've finished epoch 4!

    30

``` python
f2 = partial(show_progress, "OK I guess")
```

### Callbacks as callable classes

``` python
class ProgressShowingCallback():
    def __init__(self, exclamation="Awesome"): self.exclamation = exclamation
    def __call__(self, epoch): print(f"{self.exclamation}! We've finished epoch {epoch}!")
```

``` python
cb = ProgressShowingCallback("Just super")
```

``` python
slow_calculation(cb)
```

    Just super! We've finished epoch 0!
    Just super! We've finished epoch 1!
    Just super! We've finished epoch 2!
    Just super! We've finished epoch 3!
    Just super! We've finished epoch 4!

    30

### Multiple callback funcs; `*args` and `**kwargs`

``` python
def f(*a, **b): print(f"args: {a}, {type(a)}; kwargs: {b}")
```

``` python
f(3,'a', 1, thing2 = "asd", thing1="hello")
```

    args: (3, 'a', 1), <class 'tuple'>; kwargs: {'thing2': 'asd', 'thing1': 'hello'}

``` python
def g(a,b,c=0): print(a,b,c)
```

``` python
args = [1,2]
kwargs = {'c':3}
g(*args, **kwargs)
```

    1 2 3

``` python
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb: cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb: cb.after_calc(i, val=res)
    return res
```

``` python
class PrintStepCallback():
    def before_calc(self, *args, **kwargs): print(f"About to start")
    def after_calc (self, *args, **kwargs): print(f"Done step")
```

``` python
slow_calculation(PrintStepCallback())
```

    About to start
    Done step
    About to start
    Done step
    About to start
    Done step
    About to start
    Done step
    About to start
    Done step

    30

``` python
class PrintStatusCallback():
    def __init__(self): pass
    def before_calc(self, epoch, **kwargs): print(f"About to start: {epoch}")
    def after_calc (self, epoch, val, **kwargs): print(f"After {epoch}: {val}")
```

``` python
slow_calculation(PrintStatusCallback())
```

    About to start: 0
    After 0: 0
    About to start: 1
    After 1: 1
    About to start: 2
    After 2: 5
    About to start: 3
    After 3: 14
    About to start: 4
    After 4: 30

    30

### Modifying behavior

``` python
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb and hasattr(cb,'before_calc'): cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb and hasattr(cb,'after_calc'):
            if cb.after_calc(i, res):
                print("stopping early")
                break
    return res
```

``` python
class PrintAfterCallback():
    def after_calc (self, epoch, val):
        print(f"After {epoch}: {val}")
        if val>10: return True
```

``` python
slow_calculation(PrintAfterCallback())
```

    After 0: 0
    After 1: 1
    After 2: 5
    After 3: 14
    stopping early

    14

``` python
class SlowCalculator():
    def __init__(self, cb=None): self.cb,self.res = cb,0
    
    def callback(self, cb_name, *args):
        if not self.cb: return
        cb = getattr(self.cb,cb_name, None)
        if cb: return cb(self, *args)

    def calc(self):
        for i in range(5):
            self.callback('before_calc', i)
            self.res += i*i
            sleep(1)
            if self.callback('after_calc', i):
                print("stopping early")
                break
```

``` python
class ModifyingCallback():
    def after_calc (self, calc, epoch):
        print(f"After {epoch}: {calc.res}")
        if calc.res>10: return True
        if calc.res<3: calc.res = calc.res*2
```

``` python
calculator = SlowCalculator(ModifyingCallback())
```

``` python
calculator.calc()
calculator.res
```

    After 0: 0
    After 1: 1
    After 2: 6
    After 3: 15
    stopping early

    15

## `__dunder__` thingies

Anything that looks like `__this__` is, in some way, *special*. Python, or some library, can define some functions that they will call at certain documented times. For instance, when your class is setting up a new object, python will call `__init__`. These are defined as part of the python [data model](https://docs.python.org/3/reference/datamodel.html#object.__init__).

For instance, if python sees `+`, then it will call the special method `__add__`. If you try to display an object in Jupyter (or lots of other places in Python) it will call `__repr__`.

``` python
class SloppyAdder():
    def __init__(self,o): self.o=o
    def __add__(self,b): return SloppyAdder(self.o + b.o + 0.01)
    def __repr__(self): return str(self.o)
```

``` python
a = SloppyAdder(1)
b = SloppyAdder(2)
a+b+a
```

    4.02

Special methods you should probably know about (see data model link above) are:

- `__getitem__`
- `__getattr__`
- `__setattr__`
- `__del__`
- `__init__`
- `__new__`
- `__enter__`
- `__exit__`
- `__len__`
- `__repr__`
- `__str__`

### `__getattr__` and `getattr`

``` python
class A: 
    a,b=1,2
```

``` python
a = A()
```

``` python
a.b
```

    2

``` python
getattr(a, 'b')
```

    2

``` python
getattr(a, 'b' if random.random()>0.5 else 'a')
```

    1

``` python
class B:
    a,b=1,2
    def __getattr__(self, k):
        if k[0]=='_': raise AttributeError(k)
        return f'Hello from {k}'
```

``` python
b = B()
```

``` python
b.a
```

    1

``` python
b.foo
```

    'Hello from foo'

Back to top
