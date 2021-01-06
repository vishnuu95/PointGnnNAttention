"""This file defines classes for the graph neural network. """

from functools import partial

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def instance_normalization(features):
	with tf.variable_scope(None, default_name='IN'):
		mean, variance = tf.nn.moments(
			features, [0], name='IN_stats', keep_dims=True)
		features = tf.nn.batch_normalization(
			features, mean, variance, None, None, 1e-12, name='IN_apply')
	return(features)

normalization_fn_dict = {
	'fused_BN_center': slim.batch_norm,
	'BN': partial(slim.batch_norm, fused=False, center=False),
	'BN_center': partial(slim.batch_norm, fused=False),
	'IN': instance_normalization,
	'NONE': None
}
activation_fn_dict = {
	'ReLU': tf.nn.relu,
	'ReLU6': tf.nn.relu6,
	'LeakyReLU': partial(tf.nn.leaky_relu, alpha=0.01),
	'ELU':tf.nn.elu,
	'NONE': None,
	'Sigmoid': tf.nn.sigmoid,
	'Tanh': tf.nn.tanh,
}

def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
	is_logits=False, num_layer=4, normalization_type="fused_BN_center",
	activation_type='ReLU'):
	"""A function to create multiple layers of neural network to compute
	features passing through each edge.

	Args:
		sv: a [N, M] or [T, DEGREE, M] tensor.
		N is the total number of edges, M is the length of features. T is
		the number of recieving vertices, DEGREE is the in-degree of each
		recieving vertices. When a [T, DEGREE, M] tensor is provided, the
		degree of each recieving vertex is assumed to be same.
		N is the total number of edges, M is the length of features. T is
		the number of recieving vertices, DEGREE is the in-degree of each
		recieving vertices. When a [T, DEGREE, M] tensor is provided, the
		degree of each recieving vertex is assumed to be same.
		mask: a optional [N, 1] or [T, DEGREE, 1] tensor. A value 1 is used
		to indicate a valid output feature, while a value 0 indicates
		an invalid output feature which is set to 0.
		num_layer: number of layers to add.

	returns: a [N, K] tensor or [T, DEGREE, K].
		K is the length of the new features on the edge.
	"""
	assert len(sv.shape) == 2
	assert len(Ks) == num_layer-1
	if is_logits:
		features = sv
		for i in range(num_layer-1):
			features = slim.fully_connected(features, Ks[i],
				activation_fn=activation_fn_dict[activation_type],
				normalizer_fn=normalization_fn_dict[normalization_type],
				)
		features = slim.fully_connected(features, num_classes,
			activation_fn=None,
			normalizer_fn=None
			)
	else:
		features = sv
		for i in range(num_layer-1):
			features = slim.fully_connected(features, Ks[i],
				activation_fn=activation_fn_dict[activation_type],
				normalizer_fn=normalization_fn_dict[normalization_type],
				)
		features = slim.fully_connected(features, num_classes,
			activation_fn=activation_fn_dict[activation_type],
			normalizer_fn=normalization_fn_dict[normalization_type],
			)
	if mask is not None:
		features = features * mask
	return features

def multi_layer_neural_network_fn(features, Ks=(64, 32, 64), is_logits=False,
	normalization_type="fused_BN_center", activation_type='ReLU'):
	"""A function to create multiple layers of neural network.
	"""
	assert len(features.shape) == 2
	if is_logits:
		for i in range(len(Ks)-1):
			features = slim.fully_connected(features, Ks[i],
				activation_fn=activation_fn_dict[activation_type],
				normalizer_fn=normalization_fn_dict[normalization_type])
		features = slim.fully_connected(features, Ks[-1],
			activation_fn=None,
			normalizer_fn=None)
	else:
		for i in range(len(Ks)):
			features = slim.fully_connected(features, Ks[i],
				activation_fn=activation_fn_dict[activation_type],
				normalizer_fn=normalization_fn_dict[normalization_type])
	return features

# def graph_scatter_attn_fn(point_features, point_centers, num_centers):
# 	fout_list = [] # list of output tensors
# 	# For each key point center, apply attention and find weight mean
# 	# print("num_centers: {}".format(num_centers[0]))

# 	for k in range(num_centers[0]):
# 		# Get all feature vectors corresponsding to keypoint k
# 		feat = tf.gather(point_features, point_centers[k])
# 		print("feat_vec shape: {}".format(feat.shape))
# 		# pass all these neighbors through a neural net1  without activation fn = a1*w*h 
# 		f1 = multi_layer_neural_network_fn(feat,
# 		 normalization_type='NONE', activation_type='NONE')
# 		print("f1: {}".format(f1.shape))
# 		# pass all these neighbors through a neural net2 without activation fn = a2*w*h
# 		f2 = multi_layer_neural_network_fn(feat,
# 			normalization_type='NONE', activation_type='NONE')
# 		print("f2: {}".format(f2.shape))
# 		fn = f1 + f2 # a1*w*h + a2*w*h
# 		print("fn: {}".format(fn))
		
# 		alphas = tf.softmax(fn) # get alphas acc to GAT
# 		print("alphas: {}".format(alphas.shape))
# 		fout = tf.multiply(alphas, feat)
# 		print("fout: {}".format(fout.shape))
# 		fout_list.append(fout) # add output tensors to list
# 		print("fout_list shape: {}".format(fout_list.shape))

# 	print("stack shape: {}".format(tf.stack(fout_list, axis=0).shape))

# 	return tf.stack(fout_list, axis=0) # convert to tensor
	


def graph_scatter_max_fn(point_features, point_centers, num_centers):
	print('something!!!!!')
	print(point_features.shape)
	aggregated = tf.math.unsorted_segment_max(point_features,
		point_centers, num_centers, name='scatter_max')
	return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
	aggregated = tf.math.unsorted_segment_sum(point_features,
		point_centers, num_centers, name='scatter_sum')
	return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
	aggregated = tf.math.unsorted_segment_mean(point_features,
		point_centers, num_centers, name='scatter_mean')
	return aggregated

class ClassAwarePredictor(object):
	"""A class to predict 3D bounding boxes and class labels."""

	def __init__(self, cls_fn, loc_fn):
		"""
		Args:
			cls_fn: a function to classify labels.
			loc_fn: a function to predict 3D bounding boxes.
		"""
		self._cls_fn = cls_fn
		self._loc_fn = loc_fn

	def apply_regular(self, features, num_classes, box_encoding_len,
		normalization_type='fused_BN_center',
		activation_type='ReLU'):
		"""
		Args:
			input_v: input feature vectors. [N, M].
			output_v: not used.
			A: not used.
			num_classes: the number of classes to predict.

		returns: logits, box_encodings.
		"""
		box_encodings_list = []
		with tf.variable_scope('predictor'):
			with tf.variable_scope('cls'):
				logits = self._cls_fn(
					features, num_classes=num_classes, is_logits=True,
					normalization_type=normalization_type,
					activation_type=activation_type)
			with tf.variable_scope('loc'):
				for class_idx in range(num_classes):
					with tf.variable_scope('cls_%d' % class_idx):
						box_encodings = self._loc_fn(
							features, num_classes=box_encoding_len,
							is_logits=True,
							normalization_type=normalization_type,
							activation_type=activation_type)
						box_encodings = tf.expand_dims(box_encodings, axis=1)
						box_encodings_list.append(box_encodings)
			box_encodings = tf.concat(box_encodings_list, axis=1)
		return logits, box_encodings

class ClassAwareSeparatedPredictor(object):
	"""A class to predict 3D bounding boxes and class labels."""

	def __init__(self, cls_fn, loc_fn):
		"""
		Args:
			cls_fn: a function to classify labels.
			loc_fn: a function to predict 3D bounding boxes.
		"""
		self._cls_fn = cls_fn
		self._loc_fn = loc_fn

	def apply_regular(self, features, num_classes, box_encoding_len,
		normalization_type='fused_BN_center',
		activation_type='ReLU'):
		"""
		Args:
			input_v: input feature vectors. [N, M].
			output_v: not used.
			A: not used.
			num_classes: the number of classes to predict.

		returns: logits, box_encodings.
		"""
		box_encodings_list = []
		with tf.variable_scope('predictor'):
			with tf.variable_scope('cls'):
				logits = self._cls_fn(
					features, num_classes=num_classes, is_logits=True,
					normalization_type=normalization_type,
					activation_type=activation_type)
			features_splits = tf.split(features, num_classes, axis=-1)
			with tf.variable_scope('loc'):
				for class_idx in range(num_classes):
					with tf.variable_scope('cls_%d' % class_idx):
						box_encodings = self._loc_fn(
							features_splits[class_idx],
							num_classes=box_encoding_len,
							is_logits=True,
							normalization_type=normalization_type,
							activation_type=activation_type)
						box_encodings = tf.expand_dims(box_encodings, axis=1)
						box_encodings_list.append(box_encodings)
			box_encodings = tf.concat(box_encodings_list, axis=1)
		return logits, box_encodings

class PointSetPooling(object):
	"""A class to implement local graph netural network."""

	def __init__(self,
		point_feature_fn=multi_layer_neural_network_fn,
		aggregation_fn=graph_scatter_max_fn,
		output_fn=multi_layer_neural_network_fn):
		print("In Point set pooling")
		self._point_feature_fn = point_feature_fn
		self._aggregation_fn = aggregation_fn
		self._output_fn = output_fn

	def apply_regular(self,
		point_features,
		point_coordinates,
		keypoint_indices,
		set_indices,
		point_MLP_depth_list=None,
		point_MLP_normalization_type='fused_BN_center',
		point_MLP_activation_type = 'ReLU',
		output_MLP_depth_list=None,
		output_MLP_normalization_type='fused_BN_center',
		output_MLP_activation_type = 'ReLU'):
		"""apply a features extraction from point sets.

		Args:
			point_features: a [N, M] tensor. N is the number of points.
			M is the length of the features.
			point_coordinates: a [N, D] tensor. N is the number of points.
			D is the dimension of the coordinates.
			keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
			set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
			i.e. (i, j) indicates point[i] belongs to the point set created by
			grouping around keypoint[j].
			point_MLP_depth_list: a list of MLP units to extract point features.
			point_MLP_normalization_type: the normalization function of MLP.
			point_MLP_activation_type: the activation function of MLP.
			output_MLP_depth_list: a list of MLP units to embedd set features.
			output_MLP_normalization_type: the normalization function of MLP.
			output_MLP_activation_type: the activation function of MLP.

		returns: a [K, output_depth] tensor as the set feature.
		Output_depth depends on the feature extraction options that
		are selected.
		"""
		# Gather the points in a set
		print("In apply regular of point set pooling")
		print("Set indices: ", set_indices.shape)
		print("point_features indices: ", point_features.shape)
		# There are some points that dont belong to any set. Eliminate those
		# size = Sx1
		point_set_features = tf.gather(point_features, set_indices[:,0])

		print("point_set_features indices: ", point_set_features.shape)
		# same as above
		# size = SxD
		point_set_coordinates = tf.gather(point_coordinates, set_indices[:,0])

		print("point_set_coordinates indices: ", point_set_coordinates.shape)
		# Gather the keypoints for each set
		# 
		# Size: Sx1. Repeatedly select the same keypoint to have shape = S
		point_set_keypoint_indices = tf.gather(
			keypoint_indices, set_indices[:, 1])
		# Get the coordinates of those keypoint indices
		point_set_keypoint_coordinates = tf.gather(point_coordinates,
			point_set_keypoint_indices[:,0])
		# points within a set use relative coordinates to its keypoint ***
		point_set_coordinates = \
			point_set_coordinates - point_set_keypoint_coordinates
		# making it Sx(D+1)	
		point_set_features = tf.concat(
			[point_set_features, point_set_coordinates], axis=-1)
		with tf.variable_scope('extract_vertex_features'):
			# Step 1: Extract all vertex_features
			extracted_point_features = self._point_feature_fn(
				point_set_features,
				Ks=point_MLP_depth_list, is_logits=False,
				normalization_type=point_MLP_normalization_type,
				activation_type=point_MLP_activation_type)
			set_features = self._aggregation_fn(
				extracted_point_features, set_indices[:, 1],
				tf.shape(keypoint_indices)[0])
		with tf.variable_scope('combined_features'):
			set_features = self._output_fn(set_features,
				Ks=output_MLP_depth_list, is_logits=False,
				normalization_type=output_MLP_normalization_type,
				activation_type=output_MLP_activation_type)
		return set_features

class GraphNetAutoCenter(object):
	"""A class to implement point graph netural network layer."""

	def __init__(self,
		edge_feature_fn=multi_layer_neural_network_fn,
		aggregation_fn=graph_scatter_max_fn,
		update_fn=multi_layer_neural_network_fn,
		auto_offset_fn=multi_layer_neural_network_fn):
		self._edge_feature_fn = edge_feature_fn
		self._aggregation_fn = aggregation_fn
		self._update_fn = update_fn
		self._auto_offset_fn = auto_offset_fn

	def apply_regular(self,
		input_vertex_features,
		input_vertex_coordinates,
		NOT_USED,
		edges,
		edge_MLP_depth_list=None,
		edge_MLP_normalization_type='fused_BN_center',
		edge_MLP_activation_type = 'ReLU',
		update_MLP_depth_list=None,
		update_MLP_normalization_type='fused_BN_center',
		update_MLP_activation_type = 'ReLU',
		auto_offset=False,
		auto_offset_MLP_depth_list=None,
		auto_offset_MLP_normalization_type='fused_BN_center',
		auto_offset_MLP_feature_activation_type = 'ReLU',
		): 

		"""apply one layer graph network on a graph. .

		Args:
			input_vertex_features: a [N, M] tensor. N is the number of vertices.
			M is the length of the features.
			input_vertex_coordinates: a [N, D] tensor. N is the number of
			vertices. D is the dimension of the coordinates.
			NOT_USED: leave it here for API compatibility.
			edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
			edge_MLP_depth_list: a list of MLP units to extract edge features.
			edge_MLP_normalization_type: the normalization function of MLP.
			edge_MLP_activation_type: the activation function of MLP.
			update_MLP_depth_list: a list of MLP units to extract update
			features.
			update_MLP_normalization_type: the normalization function of MLP.
			update_MLP_activation_type: the activation function of MLP.
			auto_offset: boolean, use auto registration or not.
			auto_offset_MLP_depth_list: a list of MLP units to compute offset.
			auto_offset_MLP_normalization_type: the normalization function.
			auto_offset_MLP_feature_activation_type: the activation function.

		returns: a [N, M] tensor. Updated vertex features.
		"""
		# Gather the source vertex of the edges
		s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
		# xj
		s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])
		# [optional] Compute the coordinates offset
		if auto_offset:
			offset = self._auto_offset_fn(input_vertex_features,
				Ks=auto_offset_MLP_depth_list, is_logits=True,
				normalization_type=auto_offset_MLP_normalization_type,
				activation_type=auto_offset_MLP_feature_activation_type)
			input_vertex_coordinates = input_vertex_coordinates + offset
		
			
		# Gather the destination vertex of the edges
		d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
		# xi + dx 

		# Prepare initial edge features
		# hi, hj
		
		d_vertex_features = tf.gather(input_vertex_features, edges[:, 1]) #***********
		hi = tf.concat(
			[d_vertex_features, d_vertex_coordinates],
			 axis=-1) #*****************
		hj = tf.concat(
			[s_vertex_features, s_vertex_coordinates],
			 axis=-1) #*****************
		print("hi, hj", hi.shape, hj.shape)
		# hi = [d_vertex_coordinates; d_vertex_features]
		# hj = [s_vertex_coordinates; s_vertex_features]
		with tf.variable_scope('extract_vertex_features'):
			# Extract edge features
			hi_features_attn = self._edge_feature_fn(
				hi,
				Ks=edge_MLP_depth_list,
				is_logits=False,
				normalization_type=edge_MLP_normalization_type,
				activation_type='NONE')
			print("hi_features_attn shape", hi_features_attn.shape)
			hj_features_attn = self._edge_feature_fn(
				hj,
				Ks=edge_MLP_depth_list,
				is_logits=False,
				normalization_type=edge_MLP_normalization_type,
				activation_type='NONE')
			print("hj_features_attn shape", hj_features_attn.shape)
			# pre_eij Sx300 (a1Whi + a2Whj)
			pre_eij = hi_features_attn + hj_features_attn
			print("pre eij shape", pre_eij.shape)
			# eij Sx1
			eij = self._edge_feature_fn(
				pre_eij,
				Ks=[64,1],
				is_logits=False,
				normalization_type='fused_BN_center',
				activation_type='LeakyReLU')
			print("pre eij shape", eij.shape)
			# exp(eij)
			exp_eij = tf.math.exp(eij)
			print("exp eij shape", exp_eij.shape)
			# segment order
			seg_order, ids = tf.unique(edges[:,1])
			# np_seg_order = seg_order.eval()

			print("seg order shape", seg_order.shape)
			# calculate sum for the denominator
			exp_eij_sum_segments = tf.math.unsorted_segment_sum(exp_eij,
		edges[:,1], tf.shape(input_vertex_features)[0], name='scatter_sum')
			print("exp eij sum shape", exp_eij_sum_segments.shape)

			# k - 50
			# edges[:,1] - 90 (50 unique)
			# seg_order - 50
			exp_eij_sum_gathered = tf.gather(exp_eij_sum_segments, ids)
			# print("idx shape ", idx.shape)
			# print("exp eij sum idx", exp_eij_sum[idx].shape)
			print("exp eij sum idx", exp_eij_sum_gathered.shape)
			# alphas Sx1 (corresponding to all neighbors)
			alphas = exp_eij/exp_eij_sum_gathered
			print("alphas ", alphas.shape)
			#aggregation 
			alpha_hj = tf.math.multiply(alphas, hj_features_attn)
			print("alpha hj", alpha_hj.shape)

			aggregated_edge_features = tf.math.unsorted_segment_sum(alpha_hj,
		edges[:,1], tf.shape(input_vertex_features)[0], name='scatter_sum')
			print("aggregated_edge_features shape ", aggregated_edge_features.shape)

			ids_order = tf.argsort(seg_order)
			aggregated_edge_features_ordered = tf.gather(aggregated_edge_features, ids_order)

			with tf.variable_scope('combined_features'):
				update_features = self._update_fn(aggregated_edge_features_ordered,
					Ks=update_MLP_depth_list, is_logits=True,
					normalization_type='fused_BN_center',
					activation_type=update_MLP_activation_type)
			print("updated features shape ", aggregated_edge_features.shape)

			return update_features

						
		###########################
		# # [sj;xj+dx - xi]
		# edge_features = tf.concat(
		# 	[s_vertex_features, s_vertex_coordinates - d_vertex_coordinates],
		# 	 axis=-1)
		# # edge_feature[i] = [s ; xj - xi + dx]
		# with tf.variable_scope('extract_vertex_features'):
		# 	# Extract edge features
		# 	edge_features = self._edge_feature_fn(
		# 		edge_features,
		# 		Ks=edge_MLP_depth_list,
		# 		is_logits=False,
		# 		normalization_type=edge_MLP_normalization_type,
		# 		activation_type=edge_MLP_activation_type)
		# 	# edge_feat = sigma(W*[s ; xj - xi + dx])
		# 	# a1W(hi) + a2W(hj) 
		# 	#****  edge_feat = a1W(hi - hj) + a2W(hi - hj)
		# 	#****  alphas = softmax(exp(edge_feat))
		# 	#****
		# 	# Aggregate edge features
		# 	aggregated_edge_features = self._aggregation_fn(
		# 		edge_features,
		# 		edges[:, 1],
		# 		tf.shape(input_vertex_features)[0])
		# #**** aggregated_edge_features = 
		# # Update vertex features
		# with tf.variable_scope('combined_features'):
		# 	update_features = self._update_fn(aggregated_edge_features,
		# 		Ks=update_MLP_depth_list, is_logits=True,
		# 		normalization_type=update_MLP_normalization_type,
		# 		activation_type=update_MLP_activation_type)
		# output_vertex_features = update_features + input_vertex_features
		# return output_vertex_features
		##############################
#### 

