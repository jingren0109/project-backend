��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
�
SGD/dense_15/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_15/bias/momentum
�
.SGD/dense_15/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/dense_15/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameSGD/dense_15/kernel/momentum
�
0SGD/dense_15/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_15/kernel/momentum*
_output_shapes

:*
dtype0
�
SGD/dense_14/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_14/bias/momentum
�
.SGD/dense_14/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/bias/momentum*
_output_shapes
:*
dtype0
�
SGD/dense_14/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *-
shared_nameSGD/dense_14/kernel/momentum
�
0SGD/dense_14/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_14/kernel/momentum*
_output_shapes

: *
dtype0
�
SGD/dense_13/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_13/bias/momentum
�
.SGD/dense_13/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/bias/momentum*
_output_shapes
: *
dtype0
�
SGD/dense_13/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_13/kernel/momentum
�
0SGD/dense_13/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_13/kernel/momentum*
_output_shapes

:@ *
dtype0
�
SGD/dense_12/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_12/bias/momentum
�
.SGD/dense_12/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_12/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@*-
shared_nameSGD/dense_12/kernel/momentum
�
0SGD/dense_12/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_12/kernel/momentum*
_output_shapes

:R@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

: *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:@ *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:R@* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:R@*
dtype0
z
serving_default_input_4Placeholder*'
_output_shapes
:���������R*
dtype0*
shape:���������R
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_12390011

NoOpNoOp
�=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�=
value�=B�= B�=
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator* 
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias*
<
0
1
&2
'3
54
65
D6
E7*
<
0
1
&2
'3
54
65
D6
E7*

F0
G1
H2
I3* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Otrace_0
Ptrace_1* 

Qtrace_0
Rtrace_1* 
* 
�
Siter
	Tdecay
Ulearning_rate
Vmomentummomentum�momentum�&momentum�'momentum�5momentum�6momentum�Dmomentum�Emomentum�*

Wserving_default* 

0
1*

0
1*
	
F0* 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 

&0
'1*

&0
'1*
	
G0* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

ttrace_0
utrace_1* 

vtrace_0
wtrace_1* 
* 

50
61*

50
61*
	
H0* 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

}trace_0* 

~trace_0* 
_Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

D0
E1*

D0
E1*
	
I0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
5
0
1
2
3
4
5
6*

�0*
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
	
F0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
G0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
H0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
I0* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_12/kernel/momentumYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_12/bias/momentumWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_13/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_13/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_14/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_14/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_15/kernel/momentumYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUESGD/dense_15/bias/momentumWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcountSGD/dense_12/kernel/momentumSGD/dense_12/bias/momentumSGD/dense_13/kernel/momentumSGD/dense_13/bias/momentumSGD/dense_14/kernel/momentumSGD/dense_14/bias/momentumSGD/dense_15/kernel/momentumSGD/dense_15/bias/momentumConst*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_12390389
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcountSGD/dense_12/kernel/momentumSGD/dense_12/bias/momentumSGD/dense_13/kernel/momentumSGD/dense_13/bias/momentumSGD/dense_14/kernel/momentumSGD/dense_14/bias/momentumSGD/dense_15/kernel/momentumSGD/dense_15/bias/momentum*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_12390464��
�

f
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389699

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
e
,__inference_dropout_9_layer_call_fn_12390056

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
I
-__inference_dropout_11_layer_call_fn_12390163

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389837`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_12390011
input_4
unknown:R@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_12389665o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12390007:($
"
_user_specified_name
12390005:($
"
_user_specified_name
12390003:($
"
_user_specified_name
12390001:($
"
_user_specified_name
12389999:($
"
_user_specified_name
12389997:($
"
_user_specified_name
12389995:($
"
_user_specified_name
12389993:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4
�
�
F__inference_dense_12_layer_call_and_return_conditional_losses_12390051

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0�
"dense_12/kernel/Regularizer/L2LossL2Loss9dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0+dense_12/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�
f
-__inference_dropout_10_layer_call_fn_12390107

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389732o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_dense_14_layer_call_and_return_conditional_losses_12389748

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"dense_14/kernel/Regularizer/L2LossL2Loss9dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0+dense_14/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_15_layer_call_fn_12390189

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_12389780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12390185:($
"
_user_specified_name
12390183:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_3_12390235L
:dense_15_kernel_regularizer_l2loss_readvariableop_resource:
identity��1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_15_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_15/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
�
+__inference_dense_12_layer_call_fn_12390036

inputs
unknown:R@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12389682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12390032:($
"
_user_specified_name
12390030:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_12390219L
:dense_13_kernel_regularizer_l2loss_readvariableop_resource:@ 
identity��1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_13_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@ *
dtype0�
"dense_13/kernel/Regularizer/L2LossL2Loss9dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0+dense_13/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_13/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^dense_13/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�	
�
__inference_loss_fn_0_12390211L
:dense_12_kernel_regularizer_l2loss_readvariableop_resource:R@
identity��1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_12_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:R@*
dtype0�
"dense_12/kernel/Regularizer/L2LossL2Loss9dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0+dense_12/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_12/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^dense_12/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389837

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_12390227L
:dense_14_kernel_regularizer_l2loss_readvariableop_resource: 
identity��1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp�
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:dense_14_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0�
"dense_14/kernel/Regularizer/L2LossL2Loss9dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0+dense_14/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_14/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^dense_14/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
�
f
-__inference_dropout_11_layer_call_fn_12390158

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389765o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390129

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

g
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390124

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
I
-__inference_dropout_10_layer_call_fn_12390112

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389826`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
H
,__inference_dropout_9_layer_call_fn_12390061

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389815`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
/__inference_sequential_3_layer_call_fn_12389903
input_4
unknown:R@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389861o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12389899:($
"
_user_specified_name
12389897:($
"
_user_specified_name
12389895:($
"
_user_specified_name
12389893:($
"
_user_specified_name
12389891:($
"
_user_specified_name
12389889:($
"
_user_specified_name
12389887:($
"
_user_specified_name
12389885:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4
�9
�
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389861
input_4#
dense_12_12389806:R@
dense_12_12389808:@#
dense_13_12389817:@ 
dense_13_12389819: #
dense_14_12389828: 
dense_14_12389830:#
dense_15_12389839:
dense_15_12389841:
identity�� dense_12/StatefulPartitionedCall�1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp� dense_13/StatefulPartitionedCall�1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp� dense_14/StatefulPartitionedCall�1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp� dense_15/StatefulPartitionedCall�1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_12_12389806dense_12_12389808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12389682�
dropout_9/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389815�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_13_12389817dense_13_12389819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12389715�
dropout_10/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389826�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0dense_14_12389828dense_14_12389830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_12389748�
dropout_11/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389837�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_15_12389839dense_15_12389841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_12389780�
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_12_12389806*
_output_shapes

:R@*
dtype0�
"dense_12/kernel/Regularizer/L2LossL2Loss9dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0+dense_12/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_13_12389817*
_output_shapes

:@ *
dtype0�
"dense_13/kernel/Regularizer/L2LossL2Loss9dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0+dense_13/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_14_12389828*
_output_shapes

: *
dtype0�
"dense_14/kernel/Regularizer/L2LossL2Loss9dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0+dense_14/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_12389839*
_output_shapes

:*
dtype0�
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
12389841:($
"
_user_specified_name
12389839:($
"
_user_specified_name
12389830:($
"
_user_specified_name
12389828:($
"
_user_specified_name
12389819:($
"
_user_specified_name
12389817:($
"
_user_specified_name
12389808:($
"
_user_specified_name
12389806:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4
�
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389815

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�j
�
$__inference__traced_restore_12390464
file_prefix2
 assignvariableop_dense_12_kernel:R@.
 assignvariableop_1_dense_12_bias:@4
"assignvariableop_2_dense_13_kernel:@ .
 assignvariableop_3_dense_13_bias: 4
"assignvariableop_4_dense_14_kernel: .
 assignvariableop_5_dense_14_bias:4
"assignvariableop_6_dense_15_kernel:.
 assignvariableop_7_dense_15_bias:%
assignvariableop_8_sgd_iter:	 &
assignvariableop_9_sgd_decay: /
%assignvariableop_10_sgd_learning_rate: *
 assignvariableop_11_sgd_momentum: #
assignvariableop_12_total: #
assignvariableop_13_count: B
0assignvariableop_14_sgd_dense_12_kernel_momentum:R@<
.assignvariableop_15_sgd_dense_12_bias_momentum:@B
0assignvariableop_16_sgd_dense_13_kernel_momentum:@ <
.assignvariableop_17_sgd_dense_13_bias_momentum: B
0assignvariableop_18_sgd_dense_14_kernel_momentum: <
.assignvariableop_19_sgd_dense_14_bias_momentum:B
0assignvariableop_20_sgd_dense_15_kernel_momentum:<
.assignvariableop_21_sgd_dense_15_bias_momentum:
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_14_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_14_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_15_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_15_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_sgd_dense_12_kernel_momentumIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_sgd_dense_12_bias_momentumIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_sgd_dense_13_kernel_momentumIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_sgd_dense_13_bias_momentumIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp0assignvariableop_18_sgd_dense_14_kernel_momentumIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_sgd_dense_14_bias_momentumIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp0assignvariableop_20_sgd_dense_15_kernel_momentumIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_sgd_dense_15_bias_momentumIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp::6
4
_user_specified_nameSGD/dense_15/bias/momentum:<8
6
_user_specified_nameSGD/dense_15/kernel/momentum::6
4
_user_specified_nameSGD/dense_14/bias/momentum:<8
6
_user_specified_nameSGD/dense_14/kernel/momentum::6
4
_user_specified_nameSGD/dense_13/bias/momentum:<8
6
_user_specified_nameSGD/dense_13/kernel/momentum::6
4
_user_specified_nameSGD/dense_12/bias/momentum:<8
6
_user_specified_nameSGD/dense_12/kernel/momentum:%!

_user_specified_namecount:%!

_user_specified_nametotal:,(
&
_user_specified_nameSGD/momentum:1-
+
_user_specified_nameSGD/learning_rate:)
%
#
_user_specified_name	SGD/decay:(	$
"
_user_specified_name
SGD/iter:-)
'
_user_specified_namedense_15/bias:/+
)
_user_specified_namedense_15/kernel:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:-)
'
_user_specified_namedense_13/bias:/+
)
_user_specified_namedense_13/kernel:-)
'
_user_specified_namedense_12/bias:/+
)
_user_specified_namedense_12/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390180

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

g
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390175

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

g
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389765

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_14_layer_call_and_return_conditional_losses_12390153

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:����������
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"dense_14/kernel/Regularizer/L2LossL2Loss9dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0+dense_14/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_14/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
+__inference_dense_14_layer_call_fn_12390138

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_12389748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12390134:($
"
_user_specified_name
12390132:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
F__inference_dense_15_layer_call_and_return_conditional_losses_12389780

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_dense_13_layer_call_and_return_conditional_losses_12389715

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
"dense_13/kernel/Regularizer/L2LossL2Loss9dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0+dense_13/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

g
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389732

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:��������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
/__inference_sequential_3_layer_call_fn_12389882
input_4
unknown:R@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389803o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12389878:($
"
_user_specified_name
12389876:($
"
_user_specified_name
12389874:($
"
_user_specified_name
12389872:($
"
_user_specified_name
12389870:($
"
_user_specified_name
12389868:($
"
_user_specified_name
12389866:($
"
_user_specified_name
12389864:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4
�
�
F__inference_dense_13_layer_call_and_return_conditional_losses_12390102

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
"dense_13/kernel/Regularizer/L2LossL2Loss9dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0+dense_13/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� �
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_13/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
!__inference__traced_save_12390389
file_prefix8
&read_disablecopyonread_dense_12_kernel:R@4
&read_1_disablecopyonread_dense_12_bias:@:
(read_2_disablecopyonread_dense_13_kernel:@ 4
&read_3_disablecopyonread_dense_13_bias: :
(read_4_disablecopyonread_dense_14_kernel: 4
&read_5_disablecopyonread_dense_14_bias::
(read_6_disablecopyonread_dense_15_kernel:4
&read_7_disablecopyonread_dense_15_bias:+
!read_8_disablecopyonread_sgd_iter:	 ,
"read_9_disablecopyonread_sgd_decay: 5
+read_10_disablecopyonread_sgd_learning_rate: 0
&read_11_disablecopyonread_sgd_momentum: )
read_12_disablecopyonread_total: )
read_13_disablecopyonread_count: H
6read_14_disablecopyonread_sgd_dense_12_kernel_momentum:R@B
4read_15_disablecopyonread_sgd_dense_12_bias_momentum:@H
6read_16_disablecopyonread_sgd_dense_13_kernel_momentum:@ B
4read_17_disablecopyonread_sgd_dense_13_bias_momentum: H
6read_18_disablecopyonread_sgd_dense_14_kernel_momentum: B
4read_19_disablecopyonread_sgd_dense_14_bias_momentum:H
6read_20_disablecopyonread_sgd_dense_15_kernel_momentum:B
4read_21_disablecopyonread_sgd_dense_15_bias_momentum:
savev2_const
identity_45��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_12_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:R@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:R@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:R@z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_12_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_13_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@ z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_13_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_14_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

: z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_14_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_15_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_15_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_15_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:u
Read_8/DisableCopyOnReadDisableCopyOnRead!read_8_disablecopyonread_sgd_iter"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp!read_8_disablecopyonread_sgd_iter^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: v
Read_9/DisableCopyOnReadDisableCopyOnRead"read_9_disablecopyonread_sgd_decay"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp"read_9_disablecopyonread_sgd_decay^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead+read_10_disablecopyonread_sgd_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp+read_10_disablecopyonread_sgd_learning_rate^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_sgd_momentum"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_sgd_momentum^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_12/DisableCopyOnReadDisableCopyOnReadread_12_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpread_12_disablecopyonread_total^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_13/DisableCopyOnReadDisableCopyOnReadread_13_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpread_13_disablecopyonread_count^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead6read_14_disablecopyonread_sgd_dense_12_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp6read_14_disablecopyonread_sgd_dense_12_kernel_momentum^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:R@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:R@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:R@�
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_sgd_dense_12_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_sgd_dense_12_bias_momentum^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_16/DisableCopyOnReadDisableCopyOnRead6read_16_disablecopyonread_sgd_dense_13_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp6read_16_disablecopyonread_sgd_dense_13_kernel_momentum^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@ �
Read_17/DisableCopyOnReadDisableCopyOnRead4read_17_disablecopyonread_sgd_dense_13_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp4read_17_disablecopyonread_sgd_dense_13_bias_momentum^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead6read_18_disablecopyonread_sgd_dense_14_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp6read_18_disablecopyonread_sgd_dense_14_kernel_momentum^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_19/DisableCopyOnReadDisableCopyOnRead4read_19_disablecopyonread_sgd_dense_14_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp4read_19_disablecopyonread_sgd_dense_14_bias_momentum^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead6read_20_disablecopyonread_sgd_dense_15_kernel_momentum"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp6read_20_disablecopyonread_sgd_dense_15_kernel_momentum^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_sgd_dense_15_bias_momentum"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_sgd_dense_15_bias_momentum^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *%
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_44Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_45IdentityIdentity_44:output:0^NoOp*
T0*
_output_shapes
: �	
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_45Identity_45:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst::6
4
_user_specified_nameSGD/dense_15/bias/momentum:<8
6
_user_specified_nameSGD/dense_15/kernel/momentum::6
4
_user_specified_nameSGD/dense_14/bias/momentum:<8
6
_user_specified_nameSGD/dense_14/kernel/momentum::6
4
_user_specified_nameSGD/dense_13/bias/momentum:<8
6
_user_specified_nameSGD/dense_13/kernel/momentum::6
4
_user_specified_nameSGD/dense_12/bias/momentum:<8
6
_user_specified_nameSGD/dense_12/kernel/momentum:%!

_user_specified_namecount:%!

_user_specified_nametotal:,(
&
_user_specified_nameSGD/momentum:1-
+
_user_specified_nameSGD/learning_rate:)
%
#
_user_specified_name	SGD/decay:(	$
"
_user_specified_name
SGD/iter:-)
'
_user_specified_namedense_15/bias:/+
)
_user_specified_namedense_15/kernel:-)
'
_user_specified_namedense_14/bias:/+
)
_user_specified_namedense_14/kernel:-)
'
_user_specified_namedense_13/bias:/+
)
_user_specified_namedense_13/kernel:-)
'
_user_specified_namedense_12/bias:/+
)
_user_specified_namedense_12/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�1
�
#__inference__wrapped_model_12389665
input_4F
4sequential_3_dense_12_matmul_readvariableop_resource:R@C
5sequential_3_dense_12_biasadd_readvariableop_resource:@F
4sequential_3_dense_13_matmul_readvariableop_resource:@ C
5sequential_3_dense_13_biasadd_readvariableop_resource: F
4sequential_3_dense_14_matmul_readvariableop_resource: C
5sequential_3_dense_14_biasadd_readvariableop_resource:F
4sequential_3_dense_15_matmul_readvariableop_resource:C
5sequential_3_dense_15_biasadd_readvariableop_resource:
identity��,sequential_3/dense_12/BiasAdd/ReadVariableOp�+sequential_3/dense_12/MatMul/ReadVariableOp�,sequential_3/dense_13/BiasAdd/ReadVariableOp�+sequential_3/dense_13/MatMul/ReadVariableOp�,sequential_3/dense_14/BiasAdd/ReadVariableOp�+sequential_3/dense_14/MatMul/ReadVariableOp�,sequential_3/dense_15/BiasAdd/ReadVariableOp�+sequential_3/dense_15/MatMul/ReadVariableOp�
+sequential_3/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_12_matmul_readvariableop_resource*
_output_shapes

:R@*
dtype0�
sequential_3/dense_12/MatMulMatMulinput_43sequential_3/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,sequential_3/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_3/dense_12/BiasAddBiasAdd&sequential_3/dense_12/MatMul:product:04sequential_3/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
sequential_3/dense_12/ReluRelu&sequential_3/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
sequential_3/dropout_9/IdentityIdentity(sequential_3/dense_12/Relu:activations:0*
T0*'
_output_shapes
:���������@�
+sequential_3/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_13_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
sequential_3/dense_13/MatMulMatMul(sequential_3/dropout_9/Identity:output:03sequential_3/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
,sequential_3/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_3/dense_13/BiasAddBiasAdd&sequential_3/dense_13/MatMul:product:04sequential_3/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� |
sequential_3/dense_13/ReluRelu&sequential_3/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
 sequential_3/dropout_10/IdentityIdentity(sequential_3/dense_13/Relu:activations:0*
T0*'
_output_shapes
:��������� �
+sequential_3/dense_14/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_14_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
sequential_3/dense_14/MatMulMatMul)sequential_3/dropout_10/Identity:output:03sequential_3/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_3/dense_14/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_14/BiasAddBiasAdd&sequential_3/dense_14/MatMul:product:04sequential_3/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
sequential_3/dense_14/ReluRelu&sequential_3/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:����������
 sequential_3/dropout_11/IdentityIdentity(sequential_3/dense_14/Relu:activations:0*
T0*'
_output_shapes
:����������
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
sequential_3/dense_15/MatMulMatMul)sequential_3/dropout_11/Identity:output:03sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
IdentityIdentity&sequential_3/dense_15/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^sequential_3/dense_12/BiasAdd/ReadVariableOp,^sequential_3/dense_12/MatMul/ReadVariableOp-^sequential_3/dense_13/BiasAdd/ReadVariableOp,^sequential_3/dense_13/MatMul/ReadVariableOp-^sequential_3/dense_14/BiasAdd/ReadVariableOp,^sequential_3/dense_14/MatMul/ReadVariableOp-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 2\
,sequential_3/dense_12/BiasAdd/ReadVariableOp,sequential_3/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_12/MatMul/ReadVariableOp+sequential_3/dense_12/MatMul/ReadVariableOp2\
,sequential_3/dense_13/BiasAdd/ReadVariableOp,sequential_3/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_13/MatMul/ReadVariableOp+sequential_3/dense_13/MatMul/ReadVariableOp2\
,sequential_3/dense_14/BiasAdd/ReadVariableOp,sequential_3/dense_14/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_14/MatMul/ReadVariableOp+sequential_3/dense_14/MatMul/ReadVariableOp2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4
�
�
F__inference_dense_15_layer_call_and_return_conditional_losses_12390203

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389826

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390078

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
+__inference_dense_13_layer_call_fn_12390087

inputs
unknown:@ 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12389715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
12390083:($
"
_user_specified_name
12390081:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
F__inference_dense_12_layer_call_and_return_conditional_losses_12389682

inputs0
matmul_readvariableop_resource:R@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:R@*
dtype0�
"dense_12/kernel/Regularizer/L2LossL2Loss9dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0+dense_12/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_12/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�

f
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390073

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*
seed2*
seed���)[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�=
�
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389803
input_4#
dense_12_12389683:R@
dense_12_12389685:@#
dense_13_12389716:@ 
dense_13_12389718: #
dense_14_12389749: 
dense_14_12389751:#
dense_15_12389781:
dense_15_12389783:
identity�� dense_12/StatefulPartitionedCall�1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp� dense_13/StatefulPartitionedCall�1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp� dense_14/StatefulPartitionedCall�1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp� dense_15/StatefulPartitionedCall�1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp�"dropout_10/StatefulPartitionedCall�"dropout_11/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_12_12389683dense_12_12389685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_12_layer_call_and_return_conditional_losses_12389682�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_12389699�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_13_12389716dense_13_12389718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_12389715�
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0"^dropout_9/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_12389732�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0dense_14_12389749dense_14_12389751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_14_layer_call_and_return_conditional_losses_12389748�
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_12389765�
 dense_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_15_12389781dense_15_12389783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_12389780�
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_12_12389683*
_output_shapes

:R@*
dtype0�
"dense_12/kernel/Regularizer/L2LossL2Loss9dense_12/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_12/kernel/Regularizer/mulMul*dense_12/kernel/Regularizer/mul/x:output:0+dense_12/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_13_12389716*
_output_shapes

:@ *
dtype0�
"dense_13/kernel/Regularizer/L2LossL2Loss9dense_13/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_13/kernel/Regularizer/mulMul*dense_13/kernel/Regularizer/mul/x:output:0+dense_13/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_14_12389749*
_output_shapes

: *
dtype0�
"dense_14/kernel/Regularizer/L2LossL2Loss9dense_14/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_14/kernel/Regularizer/mulMul*dense_14/kernel/Regularizer/mul/x:output:0+dense_14/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_15_12389781*
_output_shapes

:*
dtype0�
"dense_15/kernel/Regularizer/L2LossL2Loss9dense_15/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;�
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0+dense_15/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall2^dense_12/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_13/StatefulPartitionedCall2^dense_13/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_14/StatefulPartitionedCall2^dense_14/kernel/Regularizer/L2Loss/ReadVariableOp!^dense_15/StatefulPartitionedCall2^dense_15/kernel/Regularizer/L2Loss/ReadVariableOp#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������R: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2f
1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp1dense_12/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2f
1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp1dense_13/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2f
1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp1dense_14/kernel/Regularizer/L2Loss/ReadVariableOp2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2f
1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp1dense_15/kernel/Regularizer/L2Loss/ReadVariableOp2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall:($
"
_user_specified_name
12389783:($
"
_user_specified_name
12389781:($
"
_user_specified_name
12389751:($
"
_user_specified_name
12389749:($
"
_user_specified_name
12389718:($
"
_user_specified_name
12389716:($
"
_user_specified_name
12389685:($
"
_user_specified_name
12389683:P L
'
_output_shapes
:���������R
!
_user_specified_name	input_4"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_40
serving_default_input_4:0���������R<
dense_150
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
._random_generator"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=_random_generator"
_tf_keras_layer
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias"
_tf_keras_layer
X
0
1
&2
'3
54
65
D6
E7"
trackable_list_wrapper
X
0
1
&2
'3
54
65
D6
E7"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Otrace_0
Ptrace_12�
/__inference_sequential_3_layer_call_fn_12389882
/__inference_sequential_3_layer_call_fn_12389903�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0zPtrace_1
�
Qtrace_0
Rtrace_12�
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389803
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389861�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0zRtrace_1
�B�
#__inference__wrapped_model_12389665input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
Siter
	Tdecay
Ulearning_rate
Vmomentummomentum�momentum�&momentum�'momentum�5momentum�6momentum�Dmomentum�Emomentum�"
	optimizer
,
Wserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
]trace_02�
+__inference_dense_12_layer_call_fn_12390036�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z]trace_0
�
^trace_02�
F__inference_dense_12_layer_call_and_return_conditional_losses_12390051�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z^trace_0
!:R@2dense_12/kernel
:@2dense_12/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
,__inference_dropout_9_layer_call_fn_12390056
,__inference_dropout_9_layer_call_fn_12390061�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390073
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390078�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0zgtrace_1
"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
mtrace_02�
+__inference_dense_13_layer_call_fn_12390087�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0
�
ntrace_02�
F__inference_dense_13_layer_call_and_return_conditional_losses_12390102�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zntrace_0
!:@ 2dense_13/kernel
: 2dense_13/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
ttrace_0
utrace_12�
-__inference_dropout_10_layer_call_fn_12390107
-__inference_dropout_10_layer_call_fn_12390112�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zttrace_0zutrace_1
�
vtrace_0
wtrace_12�
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390124
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390129�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0zwtrace_1
"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
}trace_02�
+__inference_dense_14_layer_call_fn_12390138�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
�
~trace_02�
F__inference_dense_14_layer_call_and_return_conditional_losses_12390153�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
!: 2dense_14/kernel
:2dense_14/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_dropout_11_layer_call_fn_12390158
-__inference_dropout_11_layer_call_fn_12390163�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390175
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390180�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
'
I0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_dense_15_layer_call_fn_12390189�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_dense_15_layer_call_and_return_conditional_losses_12390203�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:2dense_15/kernel
:2dense_15/bias
�
�trace_02�
__inference_loss_fn_0_12390211�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_12390219�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_12390227�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_12390235�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_sequential_3_layer_call_fn_12389882input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_sequential_3_layer_call_fn_12389903input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389803input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389861input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
�B�
&__inference_signature_wrapper_12390011input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
F0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_12_layer_call_fn_12390036inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_12_layer_call_and_return_conditional_losses_12390051inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dropout_9_layer_call_fn_12390056inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_dropout_9_layer_call_fn_12390061inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390073inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390078inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_13_layer_call_fn_12390087inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_13_layer_call_and_return_conditional_losses_12390102inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_10_layer_call_fn_12390107inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_10_layer_call_fn_12390112inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390124inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390129inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_14_layer_call_fn_12390138inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_14_layer_call_and_return_conditional_losses_12390153inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_11_layer_call_fn_12390158inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_11_layer_call_fn_12390163inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390175inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390180inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_15_layer_call_fn_12390189inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_15_layer_call_and_return_conditional_losses_12390203inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_12390211"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_12390219"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_12390227"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_12390235"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
,:*R@2SGD/dense_12/kernel/momentum
&:$@2SGD/dense_12/bias/momentum
,:*@ 2SGD/dense_13/kernel/momentum
&:$ 2SGD/dense_13/bias/momentum
,:* 2SGD/dense_14/kernel/momentum
&:$2SGD/dense_14/bias/momentum
,:*2SGD/dense_15/kernel/momentum
&:$2SGD/dense_15/bias/momentum�
#__inference__wrapped_model_12389665q&'56DE0�-
&�#
!�
input_4���������R
� "3�0
.
dense_15"�
dense_15����������
F__inference_dense_12_layer_call_and_return_conditional_losses_12390051c/�,
%�"
 �
inputs���������R
� ",�)
"�
tensor_0���������@
� �
+__inference_dense_12_layer_call_fn_12390036X/�,
%�"
 �
inputs���������R
� "!�
unknown���������@�
F__inference_dense_13_layer_call_and_return_conditional_losses_12390102c&'/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0��������� 
� �
+__inference_dense_13_layer_call_fn_12390087X&'/�,
%�"
 �
inputs���������@
� "!�
unknown��������� �
F__inference_dense_14_layer_call_and_return_conditional_losses_12390153c56/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
+__inference_dense_14_layer_call_fn_12390138X56/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
F__inference_dense_15_layer_call_and_return_conditional_losses_12390203cDE/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
+__inference_dense_15_layer_call_fn_12390189XDE/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390124c3�0
)�&
 �
inputs��������� 
p
� ",�)
"�
tensor_0��������� 
� �
H__inference_dropout_10_layer_call_and_return_conditional_losses_12390129c3�0
)�&
 �
inputs��������� 
p 
� ",�)
"�
tensor_0��������� 
� �
-__inference_dropout_10_layer_call_fn_12390107X3�0
)�&
 �
inputs��������� 
p
� "!�
unknown��������� �
-__inference_dropout_10_layer_call_fn_12390112X3�0
)�&
 �
inputs��������� 
p 
� "!�
unknown��������� �
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390175c3�0
)�&
 �
inputs���������
p
� ",�)
"�
tensor_0���������
� �
H__inference_dropout_11_layer_call_and_return_conditional_losses_12390180c3�0
)�&
 �
inputs���������
p 
� ",�)
"�
tensor_0���������
� �
-__inference_dropout_11_layer_call_fn_12390158X3�0
)�&
 �
inputs���������
p
� "!�
unknown����������
-__inference_dropout_11_layer_call_fn_12390163X3�0
)�&
 �
inputs���������
p 
� "!�
unknown����������
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390073c3�0
)�&
 �
inputs���������@
p
� ",�)
"�
tensor_0���������@
� �
G__inference_dropout_9_layer_call_and_return_conditional_losses_12390078c3�0
)�&
 �
inputs���������@
p 
� ",�)
"�
tensor_0���������@
� �
,__inference_dropout_9_layer_call_fn_12390056X3�0
)�&
 �
inputs���������@
p
� "!�
unknown���������@�
,__inference_dropout_9_layer_call_fn_12390061X3�0
)�&
 �
inputs���������@
p 
� "!�
unknown���������@F
__inference_loss_fn_0_12390211$�

� 
� "�
unknown F
__inference_loss_fn_1_12390219$&�

� 
� "�
unknown F
__inference_loss_fn_2_12390227$5�

� 
� "�
unknown F
__inference_loss_fn_3_12390235$D�

� 
� "�
unknown �
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389803r&'56DE8�5
.�+
!�
input_4���������R
p

 
� ",�)
"�
tensor_0���������
� �
J__inference_sequential_3_layer_call_and_return_conditional_losses_12389861r&'56DE8�5
.�+
!�
input_4���������R
p 

 
� ",�)
"�
tensor_0���������
� �
/__inference_sequential_3_layer_call_fn_12389882g&'56DE8�5
.�+
!�
input_4���������R
p

 
� "!�
unknown����������
/__inference_sequential_3_layer_call_fn_12389903g&'56DE8�5
.�+
!�
input_4���������R
p 

 
� "!�
unknown����������
&__inference_signature_wrapper_12390011|&'56DE;�8
� 
1�.
,
input_4!�
input_4���������R"3�0
.
dense_15"�
dense_15���������