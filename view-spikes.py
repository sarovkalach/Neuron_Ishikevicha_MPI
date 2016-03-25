#!/usr/bin/python
# coding: utf8
import numpy
import pylab
b=numpy.genfromtxt(open('spikes.dat','r'))
pylab.title(u'Диаграмма активности нейронов')
pylab.xlabel(u'Время, мс')
pylab.ylabel(u'Индекс нейрона')
pylab.plot(b[:,0],b[:,1],'go')
pylab.grid(True)
pylab.show()
