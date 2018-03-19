set terminal pdf enhanced
set output "10gspec.pdf"
set multiplot
set style data lines
set xrange [0:10]
set xlabel "wavenumber (cm^{-1})"
set ylabel "spectral radiance (W / m^2 / sr / cm^{-1})"
set tics textcolor rgb "white"

plot 'planck.dat' using 1:($2*300) w l lc "black" title ""

set style data histogram
set style histogram cluster gap 0
set style fill transparent solid 0.5
set tics textcolor rgb "black"

plot 'planck_results.dat' using 2:xtic(1) ti col 
