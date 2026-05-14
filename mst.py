def f(x):
    return x**2-28
def bisection(a,b,tol=0.0001,max_iter=50):
    if(f(a)*f(b)>=0):
        print("Bisection method fails.")
        return None
    print("Iter\t a\t b\t c\t f(c)")
    print("-"*50)
    for i in range(max_iter):
        c=(a+b)/2
        print(f"{i+1}\t {a:.6f}\t {b:.6f}\t {c:.6f}\t {f(c):.6f}")
        if abs(f(c))<tol:
            print("Root found:",c)
            return
        elif f(a)*f(c)<0:
            b=c
        else:
            a=c
    print("\nRoot not found using given iterations.")
a=5
b=6
bisection(a,b)