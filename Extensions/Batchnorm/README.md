

This is a parallel vectorized Batchnorm implementaion for CPU users.

This Batchnorm function can be used alternative to nn.BatchNorm1d() function

This Batchnorm implementaion uses AVX512 vectorization format, hence it is faster than the normal nn.Batchnorm1d()

-------------------------------------------------------------------------------------------------------------------------------------------------
To use this version of the Batchnorm you have to follwo these steps:

	1. Install the setup file in your python environment by this command
                'python setup.py install'

	2. After successful installation copy the "batchnorm_ext.py" file to your project folder and import the batchnorm file in your program
		'from batchnorm_ext import BatchNorm'  

	3. Use the BatchNorm in similar to the nn.BatchNorm1d
		'BatchNorm(out_feats)'

---------------------------------------------------------------------------------------------------------------------------------------------------

Example:
	
	bn_x = BatchNorm(100)
	input = torch.randn(30, 100)
	output = bn_x(input)

------------------------------------------------------------------------------------------------
	
		