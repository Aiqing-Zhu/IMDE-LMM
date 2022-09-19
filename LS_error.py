import numpy as np

xsize=18
titlesize = 25
legendsize=18
linewidth=3
hlinewidth = 1.5
def error_inte(integrator='AB', m=1):
    
    d=5
    a=np.zeros([3,8])#mean
    
    e_f=np.zeros([d,8])
    e_m=np.zeros([d,8])
    e_l=np.zeros([d,8])
    H=[]
    for i in range(5):
        H.append(0.002*2**i)
        for j in range(d):
            local = '.\\LSoutputs\\LS_'+integrator+'_{}_{}_{}'.format(m, 2**i, j)
            error=np.loadtxt(local+'\\results.txt')
            e_l[j, i] = error[1]
            e_f[j, i] = error[0]
            
    a[0,:] = e_f.mean(0)
    a[1,:] = e_m.mean(0)
    a[2,:] = e_l.mean(0)
    return a

def error_m(m=1):
    ab = error_inte(integrator='AB', m=m)
    bd = error_inte(integrator='BDF', m=m)
    print('&0.002 &', format(ab[2,0], '0.3e'), '&', format(ab[0,0], '0.3e'),'&', '---&', 
          format(bd[2,0], '0.3e'), '&', format(bd[0,0], '0.3e'),'&', r'---\\')
    
    print('&0.004 &', format(ab[2,1], '0.3e'), '&', format(ab[0,1], '0.3e'),'&', 
          format(order(ab[0,0], ab[0,1], 0.002, 0.004), '0.3f'),'&', 
          format(bd[2,1], '0.3e'), '&', format(bd[0,1], '0.3e'),'&',
          format(order(bd[0,0], bd[0,1], 0.002, 0.004), '0.3f'), r'\\')
    
    print('&0.008 &', format(ab[2,2], '0.3e'), '&', format(ab[0,2], '0.3e'),'&', 
          format(order(ab[0,1], ab[0,2], 0.004, 0.008), '0.3f'),'&', 
          format(bd[2,2], '0.3e'), '&', format(bd[0,2], '0.3e'),'&',
          format(order(bd[0,1], bd[0,2], 0.004, 0.008), '0.3f'), r'\\')

    print('&0.016 &', format(ab[2,3], '0.3e'), '&', format(ab[0,3], '0.3e'),'&', 
          format(order(ab[0,2], ab[0,3], 0.008, 0.016), '0.3f'),'&', 
          format(bd[2,3], '0.3e'), '&', format(bd[0,3], '0.3e'),'&',
          format(order(bd[0,2], bd[0,3], 0.008, 0.016), '0.3f'), r'\\')

    print('&0.032 &', format(ab[2,4], '0.3e'), '&', format(ab[0,4], '0.3e'),'&', 
          format(order(ab[0,3], ab[0,4], 0.016, 0.032), '0.3f'),'&', 
          format(bd[2,4], '0.3e'), '&', format(bd[0,4], '0.3e'),'&',
          format(order(bd[0,3], bd[0,4], 0.016, 0.032), '0.3f'), r'\\')
    
def order(e1, e2, h1, h2):
    return (np.log(e2)-np.log(e1))/(np.log(h2)-np.log(h1))
    

def AM_error():
    ab = error_inte(integrator='AM', m=2)
    bd = error_inte(integrator='AM', m=3)
    
    print('0.002 &', format(ab[2,0], '0.3e'), '&', format(ab[0,0], '0.3e'),'&', '---&', 
          format(bd[2,0], '0.3e'), '&', format(bd[0,0], '0.3e'),'&', r'---\\')
    
    print('0.004 &', format(ab[2,1], '0.3e'), '&', format(ab[0,1], '0.3e'),'&', 
          format(order(ab[0,0], ab[0,1], 0.002, 0.004), '0.3f'),'&', 
          format(bd[2,1], '0.3e'), '&', format(bd[0,1], '0.3e'),'&',
          format(order(bd[0,0], bd[0,1], 0.002, 0.004), '0.3f'), r'\\')
    
    print('0.008 &', format(ab[2,2], '0.3e'), '&', format(ab[0,2], '0.3e'),'&', 
          format(order(ab[0,1], ab[0,2], 0.004, 0.008), '0.3f'),'&', 
          format(bd[2,2], '0.3e'), '&', format(bd[0,2], '0.3e'),'&',
          format(order(bd[0,1], bd[0,2], 0.004, 0.008), '0.3f'), r'\\')

    print('0.016 &', format(ab[2,3], '0.3e'), '&', format(ab[0,3], '0.3e'),'&', 
          format(order(ab[0,2], ab[0,3], 0.008, 0.016), '0.3f'),'&', 
          format(bd[2,3], '0.3e'), '&', format(bd[0,3], '0.3e'),'&',
          format(order(bd[0,2], bd[0,3], 0.008, 0.016), '0.3f'), r'\\')

    print('0.032 &', format(ab[2,4], '0.3e'), '&', format(ab[0,4], '0.3e'),'&', 
          format(order(ab[0,3], ab[0,4], 0.016, 0.032), '0.3f'),'&', 
          format(bd[2,4], '0.3e'), '&', format(bd[0,4], '0.3e'),'&',
          format(order(bd[0,3], bd[0,4], 0.016, 0.032), '0.3f'), r'\\')    

    
if __name__=='__main__':
    error_m(m=1)
    error_m(m=2)
    error_m(m=3)
    AM_error()