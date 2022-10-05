# load compiler
FROM gcc:4.9 

# get and extract hmmer3.3.2 source code
RUN wget http://eddylab.org/software/hmmer/hmmer-3.3.2.tar.gz && tar -xvf hmmer-3.3.2.tar.gz
# get and extract master branch of modification file, copy into hmmer source code
RUN wget -v https://github.com/Larofeticus/hpc_hmmsearch/tarball/master && tar -xvf master && cp /Larofeticus-hpc_hmmsearch-*/hpc_hmmsearch.c /hmmer-3.3.2/src && ls -lh /hmmer-3.3.2/src

# build standard hmmer components
WORKDIR /hmmer-3.3.2
RUN ./configure && make

# build custommized top level application that implements hpc_hmmsearch
WORKDIR /hmmer-3.3.2/src
RUN gcc -std=gnu99 -O3 -fomit-frame-pointer -fstrict-aliasing -march=core2 -fopenmp -fPIC -msse2 -DHAVE_CONFIG_H  -I../easel -I../libdivsufsort -I../easel -I. -I. -o hpc_hmmsearch.o -c hpc_hmmsearch.c && gcc -std=gnu99 -O3 -fomit-frame-pointer -fstrict-aliasing -march=core2 -fopenmp -fPIC -msse2 -DHAVE_CONFIG_H  -L../easel -L./impl_sse -L../libdivsufsort -L. -o hpc_hmmsearch hpc_hmmsearch.o  -lhmmer -leasel -ldivsufsort     -lm

# check the right thing is there
RUN ./hpc_hmmsearch -h

# export binary path
ENV PATH "$PATH:/hmmer-3.3.2/src"
