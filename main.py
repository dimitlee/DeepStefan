from Stefan1D1P.solve_heat import run_heat
from Stefan1D1P.Stefan1D_direct import run_1d_1p_direct
from StefanIntegral.Stefan_Integral import run_stefan_integral
import timeit

start_time = timeit.default_timer()
run_stefan_integral()
elapsed = timeit.default_timer() - start_time
print(f"Total time: {elapsed:.2f}")