
This is a parallel implementaion of Embedding() for CPU users.

This Embedding function can be used alternative to nn.Embedding() function

This Embedding implementaion uses openmp parallel format, hence it is faster than the normal nn.Embedding()

-------------------------------------------------------------------------------------------------------------------------------------------------
To use this version of the Embedding() you have to follwo these steps:

	1. Install the setup file in your python environment by this command
                'python setup.py install'

	2. After successful installation copy the "batchnorm_ext.py" file to your project folder and import the batchnorm file in your program
		'from embedding_ext import EmbeddingExt'

	   if you want to use the alternative of nn.Functional.Embedding() use this
		'from embedding_ext import embedding'

	3. Use the EmbeddingExt in similar to the nn.Embedding
		'EmbeddingExt(num_embeddings, embedding_dims)'

---------------------------------------------------------------------------------------------------------------------------------------------------

Example:
	input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
	EmbeddingExt(input)
	
------------------------------------------------------------------------------------------------
	
		