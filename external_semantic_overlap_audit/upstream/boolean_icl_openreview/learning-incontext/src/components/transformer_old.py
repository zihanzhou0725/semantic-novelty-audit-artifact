import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
import numpy as np
# from transformers import TransfoXLModel, TransfoXLConfig
from src.components.attention import MultiHeadedAttention
from src.components.transformer_encoder import Encoder, EncoderLayer, EncoderLayerFFN
from src.components.positional_encodings import 	PositionalEncoding, CosineNpiPositionalEncoding, LearnablePositionalEncoding


class TransformerCLF(nn.Module):
	def __init__(self, n_dims, d_model, n_layer, n_head, dropout=0.1, pos_encode_type ='learnable', attn_type = 'standard'):
		super(TransformerCLF, self).__init__()
		self.model_type = 'SAN'
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, 10000.0)
		elif pos_encode_type == 'cosine_npi':
			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		

		self.name = f"mysand_model={d_model}_layer={n_layer}_head={n_head}"
		self.pos_encode = True
		self.pos_mask = True
		self.d_model = d_model
		self.n_dims = n_dims
		d_ffn = 2*d_model
		
		self._read_in = nn.Linear(n_dims, d_model)

		if attn_type == 'lin_attn':
			self_attn = MultiHeadedAttention(n_head, d_model, dropout, lin_attn= True)
		else:
			self_attn = MultiHeadedAttention(n_head, d_model, dropout, lin_attn= False)

		feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )
		encoder_layers = EncoderLayerFFN(d_model, self_attn, feedforward, dropout)

		self._backbone =  Encoder(encoder_layers, n_layer)

		self._read_out = nn.Linear(d_model, 1)
		# self.sigmoid = nn.Sigmoid()
		# self.softmax = nn.LogSoftmax(dim=1)

	
		# for param in self._backbone.parameters():
		# 	if param.dim() > 1:
		# 		torch.nn.init.xavier_uniform_(param)

		# layers= len(self._backbone.h)
		# for i in range(layers):
		# 	block = self._backbone.h[i]
		# 	for param in block.attn.parameters():
		# 		param.requires_grad = False
			

		# 	for param in block.mlp.parameters():
		# 		param.requires_grad = False
		
		

		# for param in self._backbone.parameters():
		#     param.requires_grad = False

		# print('Froze all Transformer attention and MLP parameters')
		# print('First and Last 10 Layernorms are tunable')
		# print('All Layernorms are tunable')
		# print('Froze all {} attention, layernorm and MLP parameters'.format(model_name))
		# print('Tunable FFNs at the beginning and end')
		# print('Tunable linear layer and FFN at the beginning and end')

		print('My Transformer {} Normal Training: All parameters are tunable'.format(attn_type))

	def init_weights(self):
		initrange = 0.1
		self._read_in.weight.data.uniform_(-initrange, initrange)
		# if sels:
		# 	self.decode()
		self._read_out.weight.data.uniform_(-initrange, initrange)
	

	def _generate_square_subsequent_mask(self, size):
		"Mask out subsequent positions."
		attn_shape = (1, size, size)
		subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs	

	def forward(self, xs, ys, inds=None):
		# input shape (xs): (batch_size, n_points, n_dims)

		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)
		embeds = self._read_in(zs)
		embeds = embeds * math.sqrt(self.d_model)
		if self.pos_encode:
			embeds= self.pos_encoder(embeds)
		# embeds shape: (batch_size, seq_len, d_model)

		src_mask = None
		if self.pos_mask:
			# mask shape: (1, seq_len, seq_len)
			src_mask = self._generate_square_subsequent_mask(embeds.size(1)).to(embeds.device) 

		output = self._backbone(embeds, src_mask)
		prediction = self._read_out(output)
		return prediction[:, ::2, 0][:, inds]  # predict only on xs

		
		




class AttCLF(nn.Module):
	def __init__(self, n_dims, d_model, n_layer, n_head, dropout=0.1, pos_encode_type ='learnable', attn_type = 'standard'):
		super(AttCLF, self).__init__()
		self.model_type = 'SAN'
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, 10000.0)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		

		self.name = f"attclf_model={d_model}_head={n_head}"
		self.pos_encode = True
		self.pos_mask = True
		self.d_model = d_model
		self.n_dims = n_dims
		self.n_layer = n_layer
		d_ffn = 2*d_model
		
		self._read_in = nn.Linear(n_dims, d_model)

		if attn_type == 'lin_attn':
			self_attn = MultiHeadedAttention(n_head, d_model, dropout, lin_attn= True)
		else:
			self_attn = MultiHeadedAttention(n_head, d_model, dropout, lin_attn= False)

		# feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )
		# encoder_layers = EncoderLayerFFN(d_model, self_attn, feedforward, dropout)
		att_block = EncoderLayer(d_model, self_attn, dropout)
		

		self._backbone =  Encoder(att_block, n_layer)


		self._read_out = nn.Linear(d_model, 1)
		# self.sigmoid = nn.Sigmoid()
		# self.softmax = nn.LogSoftmax(dim=1)

		if attn_type == 'lin_attn':
			print('Attention only Classifier linear attention Normal Training: All parameters are tunable')
		else:
			print('Attention only Classifier standard attention Normal Training: All parameters are tunable')


	def _generate_square_subsequent_mask(self, size):
		"Mask out subsequent positions."
		attn_shape = (1, size, size)
		subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

	@staticmethod
	def _combine(xs_b, ys_b):
		"""Interleaves the x's and the y's into a single sequence."""
		bsize, points, dim = xs_b.shape
		ys_b_wide = torch.cat(
			(
				ys_b.view(bsize, points, 1),
				torch.zeros(bsize, points, dim - 1, device=ys_b.device),
			),
			axis=2,
		)
		zs = torch.stack((xs_b, ys_b_wide), dim=2)
		zs = zs.view(bsize, 2 * points, dim)
		return zs	

	def forward(self, xs, ys, inds=None):
		# input shape (xs): (batch_size, n_points, n_dims)

		if inds is None:
			inds = torch.arange(ys.shape[1])
		else:
			inds = torch.tensor(inds)
			if max(inds) >= ys.shape[1] or min(inds) < 0:
				raise ValueError("inds contain indices where xs and ys are not defined")
		zs = self._combine(xs, ys)
		embeds = self._read_in(zs)
		embeds = embeds * math.sqrt(self.d_model)
		# embeds shape: (batch_size, seq_len, d_model)

		if self.pos_encode:
			embeds= self.pos_encoder(embeds)

		src_mask = None
		if self.pos_mask:
			# mask shape: (1, seq_len, seq_len)
			src_mask = self._generate_square_subsequent_mask(embeds.size(1)).to(embeds.device) 

		output = self._backbone(embeds, src_mask)
		prediction = self._read_out(output)
		return prediction[:, ::2, 0][:, inds]  # predict only on xs

		
		




# class AttCLF(nn.Module):
# 	def __init__(self, ntoken, noutputs, d_model, nhead=1, dropout=0.25, pos_encode= True, pos_encode_type ='absolute'):
# 		super(TransformerCLF, self).__init__()
# 		self.model_type = 'SAN'
# 		if pos_encode_type == 'absolute':
# 			self.pos_encoder = PositionalEncoding(d_model, dropout, 10000.0)
# 		elif pos_encode_type == 'cosine_npi':
# 			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
# 		elif pos_encode_type == 'learnable':
# 			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		
# 		self.pos_encode = pos_encode
# 		self.pos_mask = False
# 		self.d_model = d_model

# 		self.encoder= nn.Embedding(ntoken, d_model)

# 		self_attn = MultiHeadedAttention(nhead, d_model, dropout)

# 		feedforward= nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model) )
# 		encoder_layers = EncoderLayerFFN(d_model, self_attn, feedforward, dropout)

# 		self.transformer_encoder=  Encoder(encoder_layers, 1)

# 		self.decoder= nn.Linear(d_model, noutputs, bias=bias)
# 		self.sigmoid = nn.Sigmoid()
# 		self.softmax = nn.LogSoftmax(dim=1)

# 		# for p in self.parameters():
# 		# 	if p.dim() > 1:
# 		# 		nn.init.xavier_uniform(p)

# 	def init_weights(self):
# 		initrange = 0.1
# 		self.encoder.weight.data.uniform_(-initrange, initrange)
# 		if self.bias:
# 			self.decoder.bias.data.zero_()
# 		self.decoder.weight.data.uniform_(-initrange, initrange)
	

# 	def _generate_square_subsequent_mask(self, sz):
# 		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# 		mask = mask.float()
# 		mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# 		return mask
	

# 	def forward(self, src, lengths):
# 		src_mask = None
# 		if self.pos_mask:
# 			src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
		
		
# 		src = self.encoder(src) * math.sqrt(self.d_model)
# 		if self.pos_encode:
# 			src= self.pos_encoder(src)
		
# 		src = src.transpose(0,1)
# 		output= self.transformer_encoder(src, src_mask)
# 		slots = src.size(1)
# 		out_flat= output.view(-1, self.d_model)
# 		out_idxs= [(i*slots)+lengths[i].item() -1 for i in range(len(lengths))]
# 		out_vecs = out_flat[out_idxs]
# 		out = self.decoder(out_vecs)
# 		out = self.softmax(out)

		
# 		return out


















# class TransformerModel(nn.Module):
# 	"""Container module with an encoder, a recurrent or transformer module, and a decoder."""

# 	def __init__(self, ntoken, noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.5, use_embedding=False, pos_encode = True, bias = False, pos_encode_type = 'absolute', max_period = 10000.0):
# 		super(TransformerModel, self).__init__()
# 		try:
# 			from torch.nn import TransformerEncoder, TransformerEncoderLayer
# 		except:
# 			raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
# 		self.model_type = 'Transformer'
# 		self.src_mask = None

# 		# if use_embedding:
# 		# 	self.pos_encoder = PositionalEncoding(d_model, dropout)
# 		# 	self.encoder = nn.Embedding(ntoken, d_model)
# 		# else:
# 		# 	self.pos_encoder = PositionalEncoding(ntoken, dropout)
# 		# 	self.encoder = nn.Embedding(ntoken, ntoken)
# 		# 	self.encoder.weight.data =torch.eye(ntoken)
# 		# 	self.encoder.weight.requires_grad = False
# 		if pos_encode_type == 'absolute':
# 			self.pos_encoder = PositionalEncoding(d_model, dropout, max_period)
# 		elif pos_encode_type == 'cosine_npi':
# 			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
# 		elif pos_encode_type == 'learnable':
# 			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
# 		self.pos_encode = pos_encode
# 		self.encoder = nn.Embedding(ntoken, d_model)

# 		encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout)
# 		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

# 		self.d_model = d_model
# 		self.decoder = nn.Linear(d_model, noutputs, bias=bias)
# 		self.sigmoid= nn.Sigmoid()
# 		self.bias = bias

# 		self.init_weights()

# 	def _generate_square_subsequent_mask(self, sz):
# 		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# 		mask = mask.float()
# 		mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# 		return mask

# 	def init_weights(self):
# 		initrange = 0.1
# 		self.encoder.weight.data.uniform_(-initrange, initrange)
# 		if self.bias:
# 			self.decoder.bias.data.zero_()
# 		self.decoder.weight.data.uniform_(-initrange, initrange)

# 	def forward(self, src, has_mask=True, get_attns = False, get_encoder_reps = False):
# 		if has_mask:
# 			device = src.device
# 			mask = self._generate_square_subsequent_mask(len(src)).to(device)
# 			self.src_mask = mask
# 		else:
# 			self.src_mask = None
# 		src = self.encoder(src) * math.sqrt(self.d_model)
# 		if self.pos_encode:
# 			src = self.pos_encoder(src)
# 		if get_attns:
# 			attns = []
# 			encoder_layers = self.transformer_encoder.layers
# 			inp = src
# 			for layer in encoder_layers:
# 				attn = layer.self_attn(inp, inp, inp, attn_mask = self.src_mask)[1]
# 				inp = layer(inp, src_mask = self.src_mask) 
# 				attns.append(attn)


# 		transformer_output = self.transformer_encoder(src, self.src_mask)
# 		output = self.decoder(transformer_output)
# 		output = self.sigmoid(output)
# 		# return F.log_softmax(output, dim=-1)
		
# 		if get_attns:
# 			return output, attns	
		
# 		if get_encoder_reps:
# 			return output, transformer_output

# 		return output

