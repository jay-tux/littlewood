data <- read.csv('~/ugent/wetrek/bruteforce-cuda/res.txt', header=F, sep=' ')
plot(c(-2,2), c(-2,2), main="Littlewood Polynomial Roots", col='black', fg='black', bg='black', xlab='Re(x)', ylab='Im(x)')
points(data, col='cyan', pch='.')
