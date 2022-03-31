# contains the list of microbenchmark strings
# format: list of tuples (name, IR)

ir_list = [("autogen-1", """graph(%0 : Float(1, 12, 4096, 64, strides=[3145728, 64, 768, 1], requires_grad=0, device=cuda:0),
      %1 : Float(requires_grad=0, device=cuda:0)):
  %2 : int[] = prim::Constant[value=[1, 12, 64, 64, 64]]()
  %3 : Float(1, 12, 4096, 64, strides=[3145728, 64, 768, 1], requires_grad=0, device=cuda:0) = aten::div(%0, %1)
  %4 : Float(1, 12, 64, 64, 64, strides=[768, 64, 49152, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %2)
  return (%4)
"""), ("autogen-2", """graph(%0 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0),
      %1 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0),
      %6 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %10 : int):
  %11 : float = prim::Constant[value=1.0000000000000001e-05]()
  %12 : float = prim::Constant[value=0.10000000000000001]()
  %13 : bool = prim::Constant[value=0]()
  %14 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0), %15 : Tensor, %16 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %13, %12, %11)
  %17 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0), %18 : Tensor, %19 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %13, %12, %11)
  %20 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0) = aten::add(%17, %14, %10)
  %21 : Float(8, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0) = aten::relu(%20)
  return (%21)
"""), ("batchnorm-silu", """graph(%0 : Float(32, 480, 14, 14, strides=[94080, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(480, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(480, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(480, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(480, strides=[1], requires_grad=0, device=cuda:0)):
  %5 : float = prim::Constant[value=1.0000000000000001e-05]()
  %6 : float = prim::Constant[value=0.10000000000000001]()
  %7 : bool = prim::Constant[value=0]()
  %8 : Float(32, 480, 14, 14, strides=[94080, 196, 14, 1], requires_grad=0, device=cuda:0), %9 : Tensor, %10 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %7, %6, %5)
  %11 : Float(32, 480, 14, 14, strides=[94080, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::silu(%8)
  return (%11)
""")]
