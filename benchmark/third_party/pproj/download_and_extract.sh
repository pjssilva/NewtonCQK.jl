#!/usr/bin/bash

# Download and extract PPROJ
if [ ! -f "PPROJ-1.0.tar_.gz" ]; then
    wget https://people.clas.ufl.edu/hager/files/PPROJ-1.0.tar_.gz
fi
tar -zxvf PPROJ-1.0.tar_.gz

# Download old compatible SuiteSparse and Metis 4, required by PPROJ
if [ ! -f "PPROJ-1.0/v4.4.7.tar.gz" ]; then
    (cd PPROJ-1.0 && wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v4.4.7.tar.gz)
fi
(cd PPROJ-1.0 && tar -zxvf v4.4.7.tar.gz)

if [ ! -f "PPROJ-1.0/SuiteSparse-4.4.7/metis-4.0.3.tar.gz" ]; then
    (cd PPROJ-1.0/SuiteSparse-4.4.7 && wget https://papers.karypis.org/glaros/files/sw/metis/metis-4.0.3.tar.gz)
fi
(cd PPROJ-1.0/SuiteSparse-4.4.7 && rm -rdf metis-4.0)
(cd PPROJ-1.0/SuiteSparse-4.4.7 && tar -zxvf metis-4.0.3.tar.gz)
(cd PPROJ-1.0/SuiteSparse-4.4.7 && mv metis-4.0.3 metis-4.0)

# Adjust Makefile in PPROJ
(cd PPROJ-1.0/Check && sed 's|/home/data/SuiteSparse/|../../$(SUITESPARSEDIR)/|g' Makefile > Makefile.new)
(cd PPROJ-1.0/Check && mv Makefile Makefile.old)
(cd PPROJ-1.0/Check && mv Makefile.new Makefile)
