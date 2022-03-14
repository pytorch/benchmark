# contains the list of microbenchmark strings
# format: list of tuples (name, IR)

ir_list = [("autogen-0", """graph(%0 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(4096, 512, strides=[512, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : int[] = prim::Constant[value=[4096, 512]]()
  %4 : int[] = prim::Constant[value=[1, 4096, 512]]()
  %5 : Float(1, 4096, 512, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %4)
  %6 : Float(1, 4096, 512, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%5, %0, %2)
  %7 : Float(4096, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%6, %3)
  %8 : Float(1, 4096, 512, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %4)
  %9 : Float(1, 4096, 512, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0) = aten::relu(%8)
  %10 : Float(4096, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %3)
  return (%10)
"""), ("autogen-1", """graph(%0 : Float(1, 12, 4096, 64, strides=[3145728, 64, 768, 1], requires_grad=0, device=cuda:0),
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
"""), ("autogen-3", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %7 : Double(requires_grad=0, device=cuda:0),
      %8 : Double(requires_grad=0, device=cuda:0),
      %9 : Double(requires_grad=0, device=cuda:0),
      %10 : Double(requires_grad=0, device=cuda:0),
      %11 : Double(requires_grad=0, device=cuda:0),
      %12 : Double(requires_grad=0, device=cuda:0),
      %13 : Double(requires_grad=0, device=cuda:0),
      %14 : Double(requires_grad=0, device=cuda:0),
      %15 : Double(requires_grad=0, device=cuda:0),
      %16 : Double(requires_grad=0, device=cuda:0),
      %17 : Double(requires_grad=0, device=cuda:0),
      %18 : Double(requires_grad=0, device=cuda:0),
      %19 : Double(requires_grad=0, device=cuda:0),
      %20 : Double(requires_grad=0, device=cuda:0),
      %21 : Double(requires_grad=0, device=cuda:0),
      %22 : Double(requires_grad=0, device=cuda:0),
      %23 : Double(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0),
      %24 : Double(requires_grad=0, device=cuda:0),
      %25 : Double(requires_grad=0, device=cuda:0),
      %26 : Double(requires_grad=0, device=cuda:0),
      %27 : int,
      %28 : int,
      %29 : int,
      %30 : int,
      %31 : int,
      %32 : int,
      %33 : int,
      %34 : int,
      %35 : int,
      %36 : int,
      %37 : int,
      %38 : int,
      %39 : int,
      %40 : int,
      %41 : int,
      %42 : int,
      %43 : int,
      %44 : int,
      %45 : int,
      %46 : int,
      %47 : int,
      %48 : int,
      %49 : int,
      %50 : int,
      %51 : int,
      %52 : int,
      %53 : int,
      %54 : int,
      %55 : int,
      %56 : int,
      %57 : int,
      %58 : int,
      %59 : int,
      %60 : int,
      %61 : int,
      %62 : int,
      %63 : int,
      %64 : int,
      %65 : int,
      %66 : int):
  %67 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %16)
  %68 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%67, %12)
  %69 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %26)
  %70 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %25)
  %71 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%70, %10, %66)
  %72 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %71)
  %73 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%72, %24, %65)
  %74 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%73, %69, %64)
  %75 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%74, %23)
  %76 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %22)
  %77 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%76, %9, %63)
  %78 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %77)
  %79 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%78, %8, %62)
  %80 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %79)
  %81 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%80, %21, %61)
  %82 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sqrt(%6)
  %83 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%82, %81)
  %84 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %20)
  %85 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%84, %7, %60)
  %86 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %85)
  %87 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%86, %19, %59)
  %88 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%87, %83, %58)
  %89 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %88)
  %90 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %18)
  %91 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%90, %5, %57)
  %92 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %91)
  %93 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%92, %3, %56)
  %94 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %93)
  %95 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%94, %17, %55)
  %96 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%95, %89, %54)
  %97 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%96, %74)
  %98 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %16)
  %99 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%98, %15, %53)
  %100 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %99)
  %101 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%100, %12)
  %102 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %14)
  %103 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%102, %13, %52)
  %104 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %103)
  %105 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%104, %12)
  %106 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%105, %11, %51)
  %107 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%106, %101, %50)
  %108 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::pow(%107, %49)
  %109 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%108, %97, %48)
  %110 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sqrt(%109)
  %111 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%105, %11, %47)
  %112 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%111, %101, %46)
  %113 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%112, %110, %45)
  %114 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%113, %75, %44)
  %115 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%114)
  %116 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%115, %2)
  %117 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%105, %11, %43)
  %118 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%117, %101, %42)
  %119 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%118, %110, %41)
  %120 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%119)
  %121 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%120, %2)
  %122 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%110, %121)
  %123 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%122, %116)
  %124 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%75, %0)
  %125 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%124, %123)
  %126 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%125, %2, %40)
  %127 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%70, %0)
  %128 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%127, %10, %39)
  %129 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%76, %0)
  %130 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%129, %9, %38)
  %131 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %130)
  %132 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%78, %8, %37)
  %133 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%132, %131, %36)
  %134 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%82, %133)
  %135 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%84, %0)
  %136 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%135, %7, %35)
  %137 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%136, %134, %34)
  %138 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %137)
  %139 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%90, %0)
  %140 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%139, %5, %33)
  %141 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %140)
  %142 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%92, %3, %32)
  %143 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%142, %141, %31)
  %144 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%143, %138, %30)
  %145 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%144, %74)
  %146 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%102, %2)
  %147 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%146, %1, %29)
  %148 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%147, %68, %28)
  %149 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%107, %0)
  %150 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%149, %148)
  %151 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%150, %145, %27)
  return (%151, %148, %146, %144, %128, %126, %123, %121, %119, %116, %114, %110, %109, %108, %107, %104, %102, %100, %96, %82, %75, %74, %68, %67)
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
"""), ("autogen-4", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : float = prim::Constant[value=9.9999999999999998e-13]()
  %8 : int[] = prim::Constant[value=[768]]()
  %9 : int[] = prim::Constant[value=[4096, 768]]()
  %10 : int[] = prim::Constant[value=[1, 4096, 768]]()
  %11 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %10)
  %12 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %3, %6)
  %13 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %9)
  %14 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%14, %2, %5)
  %16 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0), %17 : Tensor, %18 : Tensor = aten::native_layer_norm(%15, %8, %0, %1, %7)
  %19 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %9)
  return (%19)
"""), ("autogen-5", """graph(%0 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0),
      %1 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0),
      %6 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %10 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0),
      %11 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %12 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %13 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %14 : Float(160, strides=[1], requires_grad=0, device=cuda:0),
      %15 : int,
      %16 : int):
  %17 : float = prim::Constant[value=1.0000000000000001e-05]()
  %18 : float = prim::Constant[value=0.10000000000000001]()
  %19 : bool = prim::Constant[value=0]()
  %20 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0), %21 : Tensor, %22 : Tensor = aten::native_batch_norm(%10, %11, %12, %13, %14, %19, %18, %17)
  %23 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0), %24 : Tensor, %25 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %19, %18, %17)
  %26 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0), %27 : Tensor, %28 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %19, %18, %17)
  %29 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::add(%26, %23, %16)
  %30 : Float(96, 160, 7, 7, strides=[7840, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::add(%29, %20, %15)
  return (%30)
"""), ("autogen-6", """graph(%0 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : Double(requires_grad=0, device=cuda:0),
      %7 : Double(requires_grad=0, device=cuda:0),
      %8 : Double(requires_grad=0, device=cuda:0),
      %9 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %10 : Double(requires_grad=0, device=cuda:0),
      %11 : Double(requires_grad=0, device=cuda:0),
      %12 : Double(requires_grad=0, device=cuda:0),
      %13 : Double(requires_grad=0, device=cuda:0),
      %14 : Double(requires_grad=0, device=cuda:0),
      %15 : Double(requires_grad=0, device=cuda:0),
      %16 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %17 : Double(requires_grad=0, device=cuda:0),
      %18 : Double(requires_grad=0, device=cuda:0),
      %19 : Double(requires_grad=0, device=cuda:0),
      %20 : Double(requires_grad=0, device=cuda:0),
      %21 : Double(requires_grad=0, device=cuda:0),
      %22 : Double(requires_grad=0, device=cuda:0),
      %23 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %24 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %25 : Double(requires_grad=0, device=cuda:0),
      %26 : Double(requires_grad=0, device=cuda:0),
      %27 : Double(requires_grad=0, device=cuda:0),
      %28 : Double(requires_grad=0, device=cuda:0),
      %29 : Double(requires_grad=0, device=cuda:0),
      %30 : Double(requires_grad=0, device=cuda:0),
      %31 : Double(requires_grad=0, device=cuda:0),
      %32 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %33 : Double(requires_grad=0, device=cuda:0),
      %34 : Double(requires_grad=0, device=cuda:0),
      %35 : Double(requires_grad=0, device=cuda:0),
      %36 : Double(requires_grad=0, device=cuda:0),
      %37 : Double(requires_grad=0, device=cuda:0),
      %38 : Double(requires_grad=0, device=cuda:0),
      %39 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %40 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %41 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %42 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %43 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %44 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %45 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %46 : Double(requires_grad=0, device=cuda:0),
      %47 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %48 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %49 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %50 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %51 : Double(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0),
      %52 : int,
      %53 : int,
      %54 : int,
      %55 : int,
      %56 : int,
      %57 : int,
      %58 : int,
      %59 : int,
      %60 : int,
      %61 : int,
      %62 : int,
      %63 : int,
      %64 : int,
      %65 : int,
      %66 : int,
      %67 : int,
      %68 : int,
      %69 : int,
      %70 : int,
      %71 : int,
      %72 : int,
      %73 : int,
      %74 : int,
      %75 : int,
      %76 : int,
      %77 : int,
      %78 : int,
      %79 : int,
      %80 : int,
      %81 : int,
      %82 : int,
      %83 : int,
      %84 : int,
      %85 : int,
      %86 : int,
      %87 : int,
      %88 : int,
      %89 : int,
      %90 : int,
      %91 : int,
      %92 : int,
      %93 : int,
      %94 : int):
  %95 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%50, %51)
  %96 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%24, %50)
  %97 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%49, %96, %94)
  %98 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%47)
  %99 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%98, %3)
  %100 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%99, %97)
  %101 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%100, %22)
  %102 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%45, %46, %93)
  %103 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%102, %44, %92)
  %104 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%103, %101, %91)
  %105 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%104, %95, %90)
  %106 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::pow(%48, %89)
  %107 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%42, %47)
  %108 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%107, %22)
  %109 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%108, %41)
  %110 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%109, %106)
  %111 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%110, %105)
  %112 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%45, %46, %88)
  %113 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%112, %44, %87)
  %114 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%113, %101, %86)
  %115 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::pow(%43, %85)
  %116 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%108, %115)
  %117 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%116, %40)
  %118 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%117, %114)
  %119 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%42, %99)
  %120 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%119, %41)
  %121 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%120, %40)
  %122 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%121, %97)
  %123 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%95, %22)
  %124 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%123, %39)
  %125 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%124, %122, %84)
  %126 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%125, %118, %83)
  %127 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%126, %111, %82)
  %128 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%2)
  %129 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%128, %3)
  %130 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %38)
  %131 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %37)
  %132 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%131, %36, %81)
  %133 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%132, %130, %80)
  %134 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %133)
  %135 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%134, %35, %79)
  %136 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%135, %22)
  %137 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%136, %23)
  %138 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %34)
  %139 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%138, %33, %78)
  %140 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%139, %24)
  %141 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%139, %32)
  %142 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%141, %31)
  %143 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%142, %129)
  %144 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%143, %140, %77)
  %145 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%144, %137, %76)
  %146 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%145, %129)
  %147 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %30)
  %148 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%147, %29, %75)
  %149 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %148)
  %150 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %28)
  %151 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%150, %27, %74)
  %152 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %151)
  %153 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%152, %26, %73)
  %154 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %153)
  %155 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%154, %25, %72)
  %156 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%155, %149, %71)
  %157 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%156, %146, %70)
  %158 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%157, %23)
  %159 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%135, %24)
  %160 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%23, %129)
  %161 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%140, %22)
  %162 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%161, %160)
  %163 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%162, %159, %69)
  %164 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%163, %129)
  %165 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %21)
  %166 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%165, %20, %68)
  %167 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %166)
  %168 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%167, %19, %67)
  %169 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %168)
  %170 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%169, %18, %66)
  %171 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %170)
  %172 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%171, %17, %65)
  %173 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%16, %172)
  %174 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %15)
  %175 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %14)
  %176 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%175, %13, %64)
  %177 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %176)
  %178 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%177, %12, %63)
  %179 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %178)
  %180 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%179, %11, %62)
  %181 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %180)
  %182 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%181, %10, %61)
  %183 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%182, %174, %60)
  %184 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%183, %173, %59)
  %185 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %184)
  %186 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %8)
  %187 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%186, %7, %58)
  %188 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %187)
  %189 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%188, %6, %57)
  %190 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %189)
  %191 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%190, %5, %56)
  %192 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %191)
  %193 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%192, %3, %55)
  %194 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%193, %185, %54)
  %195 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%194, %164, %53)
  %196 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%195, %2)
  %197 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%196, %158, %52)
  %198 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%197, %129)
  %199 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%198, %1)
  %200 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%199, %99)
  %201 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%200, %127)
  %202 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%201, %0)
  return (%202, %198, %197, %195, %190, %188, %186, %179, %177, %175, %169, %167, %165, %163, %161, %160, %157, %152, %150, %145, %142, %139, %136, %135, %134, %131, %130, %129, %99, %97, %95)
"""), ("autogen-7", """graph(%0 : Float(8, 197, 6, 64, strides=[75648, 64, 12608, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[1576, 384]]()
  %2 : int[] = prim::Constant[value=[8, 197, 384]]()
  %3 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %4 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %1)
  return (%4)
"""), ("autogen-8", """graph(%0 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %2 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %3 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %4 : Double(requires_grad=0, device=cuda:0),
      %5 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0)):
  %6 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::log(%5)
  %7 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%3, %4)
  %8 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%7, %2)
  %9 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%8, %1)
  %10 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %6)
  %11 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %0)
  return (%11, %6)
"""), ("autogen-9", """graph(%0 : Float(1, 12, 1, 64, 64, strides=[768, 64, 49152, 768, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[1, 12, 64, 64, 1, 1]]()
  %2 : int[] = prim::Constant[value=[1, 12, 64, 64, 1]]()
  %3 : int[] = prim::Constant[value=[1, 12, 64, 64]]()
  %4 : Float(1, 12, 64, 64, strides=[768, 64, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %3)
  %5 : Float(1, 12, 64, 64, 1, strides=[768, 64, 768, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %2)
  %6 : Float(1, 12, 64, 64, 1, 1, strides=[768, 64, 768, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %1)
  return (%6, %4)
"""), ("autogen-10", """graph(%0 : Long(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Long(200, 200, strides=[200, 1], requires_grad=0, device=cuda:0)):
  %2 : int[] = prim::Constant[value=[200, 200, 1]]()
  %3 : Long(200, 200, 1, strides=[200, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %2)
  %4 : Bool(200, 200, 26, strides=[5200, 26, 1], requires_grad=0, device=cuda:0) = aten::ge(%0, %3)
  return (%4)
"""), ("autogen-11", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : int[] = prim::Constant[value=[1, 512, 12, 64]]()
  %4 : int[] = prim::Constant[value=[512, 768]]()
  %5 : int[] = prim::Constant[value=[1, 512, 768]]()
  %6 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %5)
  %7 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%6, %0, %2)
  %8 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %4)
  %9 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %5)
  %10 : Float(1, 512, 12, 64, strides=[393216, 768, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %3)
  return (%10)
"""), ("autogen-12", """graph(%0 : Float(32, 360, 14, 14, strides=[70560, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(32, 360, 1, 1, strides=[360, 1, 1, 1], requires_grad=0, device=cuda:0)):
  %2 : Float(32, 360, 1, 1, strides=[360, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::sigmoid(%1)
  %3 : Float(32, 360, 14, 14, strides=[70560, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::mul(%0, %2)
  return (%3)
"""), ("autogen-13", """graph(%0 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(32, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0),
      %5 : Float(256, strides=[1], requires_grad=0, device=cuda:0)):
  %6 : float = prim::Constant[value=1.0000000000000001e-05]()
  %7 : float = prim::Constant[value=0.10000000000000001]()
  %8 : bool = prim::Constant[value=0]()
  %9 : int[] = prim::Constant[value=[1, 256, 1, 1]]()
  %10 : Float(1, 256, 1, 1, strides=[256, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %9)
  %11 : Float(32, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0) = aten::div(%4, %10)
  %12 : Float(32, 256, 56, 56, strides=[802816, 3136, 56, 1], requires_grad=0, device=cuda:0), %13 : Tensor, %14 : Tensor = aten::native_batch_norm(%11, %0, %1, %2, %3, %8, %7, %6)
  return (%12, %13, %14)
"""), ("autogen-14", """graph(%0 : Float(8, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0),
      %1 : Float(8, 2048, 2048, strides=[1, 16384, 8], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 1, 2048, 2048, strides=[4194304, 4194304, 2048, 1], requires_grad=0, device=cuda:0),
      %5 : Float(1, 1, 1, 2048, strides=[2048, 2048, 2048, 1], requires_grad=0, device=cuda:0),
      %6 : int,
      %7 : int,
      %8 : int):
  %9 : bool = prim::Constant[value=0]()
  %10 : int = prim::Constant[value=-1]()
  %11 : int[] = prim::Constant[value=[8, 2048, 2048]]()
  %12 : int[] = prim::Constant[value=[1, 8, 2048, 2048]]()
  %13 : Float(1, 1, 2048, 2048, strides=[4194304, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %5)
  %14 : Float(1, 1, 2048, 2048, strides=[4194304, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %13, %8)
  %15 : Float(1, 1, 2048, 2048, strides=[4194304, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %2)
  %16 : Float(1, 8, 2048, 2048, strides=[8, 1, 16384, 8], requires_grad=0, device=cuda:0) = aten::reshape(%1, %12)
  %17 : Float(1, 8, 2048, 2048, strides=[8, 1, 16384, 8], requires_grad=0, device=cuda:0) = aten::add(%16, %15, %7)
  %18 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %12)
  %19 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::add(%18, %17, %6)
  %20 : Float(8, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%19, %11)
  %21 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%20, %12)
  %22 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%21, %10, %9)
  return (%22, %17)
"""), ("batchnorm-silu-mean", """graph(%0 : Float(32, 240, 14, 14, strides=[47040, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(240, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(240, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(240, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(240, strides=[1], requires_grad=0, device=cuda:0)):
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=1]()
  %7 : int[] = prim::Constant[value=[2, 3]]()
  %8 : float = prim::Constant[value=1.0000000000000001e-05]()
  %9 : float = prim::Constant[value=0.10000000000000001]()
  %10 : bool = prim::Constant[value=0]()
  %11 : Float(32, 240, 14, 14, strides=[47040, 196, 14, 1], requires_grad=0, device=cuda:0), %12 : Tensor, %13 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %10, %9, %8)
  %14 : Float(32, 240, 14, 14, strides=[47040, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::silu(%11)
  %15 : Float(32, 240, 1, 1, strides=[240, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%14, %7, %6, %5)
  return (%15, %14)
"""), ("autogen-15", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : int[] = prim::Constant[value=[512, 768]]()
  %4 : int[] = prim::Constant[value=[1, 512, 768]]()
  %5 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %4)
  %6 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%5, %0, %2)
  %7 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%6, %3)
  %8 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %4)
  %9 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %3)
  return (%9, %8)
"""), ("autogen-16", """graph(%0 : Float(1, 1, 512, 512, strides=[262144, 262144, 512, 1], requires_grad=0, device=cuda:0),
      %1 : Float(12, 512, 512, strides=[262144, 512, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : bool = prim::Constant[value=0]()
  %4 : int = prim::Constant[value=-1]()
  %5 : int[] = prim::Constant[value=[12, 512, 512]]()
  %6 : int[] = prim::Constant[value=[1, 12, 512, 512]]()
  %7 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %6)
  %8 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%7, %0, %2)
  %9 : Float(12, 512, 512, strides=[262144, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %5)
  %10 : Float(12, 512, 512, strides=[262144, 512, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%9, %4, %3)
  return (%10)
"""), ("autogen-17", """graph(%0 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0),
      %1 : Float(768, 64, 128, strides=[8192, 128, 1], requires_grad=0, device=cuda:0),
      %2 : int,
      %3 : int,
      %4 : int):
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=1]()
  %7 : int[] = prim::Constant[value=[-1]]()
  %8 : int[] = prim::Constant[value=[1, 12, 64, 64, 128]]()
  %9 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %8)
  %10 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%9, %0, %4)
  %11 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::exp(%10)
  %12 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::sum(%11, %7, %6, %5)
  %13 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::log(%12)
  %14 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%13, %0, %3)
  %15 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%9, %14, %2)
  %16 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::exp(%15)
  return (%16)
"""), ("autogen-18", """graph(%0 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0),
      %3 : Float(1, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0),
      %4 : int):
  %5 : int[] = prim::Constant[value=[1576, 384]]()
  %6 : float = prim::Constant[value=9.9999999999999995e-07]()
  %7 : int[] = prim::Constant[value=[384]]()
  %8 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %3, %4)
  %9 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0), %10 : Tensor, %11 : Tensor = aten::native_layer_norm(%8, %7, %0, %1, %6)
  %12 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %5)
  return (%12, %8)
"""), ("autogen-19", """graph(%0 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 4096, 256, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0),
      %3 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0),
      %4 : int):
  %5 : int[] = prim::Constant[value=[4096, 256]]()
  %6 : float = prim::Constant[value=9.9999999999999998e-13]()
  %7 : int[] = prim::Constant[value=[256]]()
  %8 : int[] = prim::Constant[value=[1, 4096, 256]]()
  %9 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %8)
  %10 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %9, %4)
  %11 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0), %12 : Tensor, %13 : Tensor = aten::native_layer_norm(%10, %7, %0, %1, %6)
  %14 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %5)
  return (%14, %10)
"""), ("autogen-20", """graph(%0 : Float(16, 512, 7, 7, strides=[25088, 49, 7, 1], requires_grad=0, device=cuda:0),
      %1 : Float(16, 512, 7, 7, strides=[25088, 49, 7, 1], requires_grad=0, device=cuda:0),
      %2 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %6 : int):
  %7 : int[] = prim::Constant[value=[16, 512]]()
  %8 : NoneType = prim::Constant()
  %9 : bool = prim::Constant[value=1]()
  %10 : int[] = prim::Constant[value=[-1, -2]]()
  %11 : float = prim::Constant[value=1.0000000000000001e-05]()
  %12 : float = prim::Constant[value=0.10000000000000001]()
  %13 : bool = prim::Constant[value=0]()
  %14 : Float(16, 512, 7, 7, strides=[25088, 49, 7, 1], requires_grad=0, device=cuda:0), %15 : Tensor, %16 : Tensor = aten::native_batch_norm(%1, %2, %3, %4, %5, %13, %12, %11)
  %17 : Float(16, 512, 7, 7, strides=[25088, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::add(%14, %0, %6)
  %18 : Float(16, 512, 7, 7, strides=[25088, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::relu(%17)
  %19 : Float(16, 512, 1, 1, strides=[512, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%18, %10, %9, %8)
  %20 : Float(16, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%19, %7)
  return (%20)
"""), ("autogen-21", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0),
      %4 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[512, 768]]()
  %8 : float = prim::Constant[value=9.9999999999999998e-13]()
  %9 : int[] = prim::Constant[value=[768]]()
  %10 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%3, %4, %6)
  %11 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%10, %2, %5)
  %12 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0), %13 : Tensor, %14 : Tensor = aten::native_layer_norm(%11, %9, %0, %1, %8)
  %15 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %7)
  return (%15, %12)
"""), ("autogen-22", """graph(%0 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %4 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %5 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %6 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %7 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %8 : Double(requires_grad=0, device=cuda:0),
      %9 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %10 : Double(requires_grad=0, device=cuda:0),
      %11 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %12 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %13 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %14 : Double(requires_grad=0, device=cuda:0),
      %15 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %16 : Double(requires_grad=0, device=cuda:0),
      %17 : Double(requires_grad=0, device=cuda:0),
      %18 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %19 : int,
      %20 : int,
      %21 : int,
      %22 : int,
      %23 : int,
      %24 : int,
      %25 : int,
      %26 : int,
      %27 : int):
  %28 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::pow(%18, %27)
  %29 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%28)
  %30 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%29, %17)
  %31 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%15, %16)
  %32 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %14)
  %33 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%30, %12)
  %34 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%0, %7)
  %35 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%32, %0)
  %36 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%11, %8)
  %37 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%36, %10, %26)
  %38 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%37, %9, %25)
  %39 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%38, %8)
  %40 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %4)
  %41 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %7)
  %42 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%3, %5)
  %43 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%3, %4)
  %44 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%43, %2)
  %45 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%44, %34)
  %46 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%35, %45, %24)
  %47 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%1, %33)
  %48 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%46, %47, %23)
  %49 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%48, %31, %22)
  %50 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%49, %42, %21)
  %51 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%50, %40, %20)
  %52 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%51, %41, %19)
  %53 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%52, %0)
  return (%53, %43, %42, %38, %36, %34, %33, %31, %30)
"""), ("autogen-23", """graph(%0 : Float(32, 2, 256, 28, 28, strides=[401408, 200704, 784, 28, 1], requires_grad=0, device=cuda:0),
      %1 : Float(32, 2, 1, 256, strides=[512, 256, 512, 1], requires_grad=0, device=cuda:0)):
  %2 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1]]()
  %4 : int[] = prim::Constant[value=[32, 2, 256, 1, 1]]()
  %5 : int[] = prim::Constant[value=[32, 512, 1, 1]]()
  %6 : int[] = prim::Constant[value=[32, 512]]()
  %7 : bool = prim::Constant[value=0]()
  %8 : int = prim::Constant[value=1]()
  %9 : Float(32, 2, 1, 256, strides=[512, 256, 256, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%1, %8, %7)
  %10 : Float(32, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %6)
  %11 : Float(32, 512, 1, 1, strides=[512, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%10, %5)
  %12 : Float(32, 2, 256, 1, 1, strides=[512, 256, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %4)
  %13 : Float(32, 2, 256, 28, 28, strides=[401408, 200704, 784, 28, 1], requires_grad=0, device=cuda:0) = aten::mul(%0, %12)
  %14 : Float(32, 256, 28, 28, strides=[200704, 784, 28, 1], requires_grad=0, device=cuda:0) = aten::sum(%13, %3, %7, %2)
  return (%14)
"""), ("autogen-24", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1024, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int,
      %7 : float):
  %8 : int[] = prim::Constant[value=[1024, 3072]]()
  %9 : int[] = prim::Constant[value=[1, 1024, 3072]]()
  %10 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %9)
  %11 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::pow(%10, %7)
  %12 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%11, %3)
  %13 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%10, %12, %6)
  %14 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %2)
  %15 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::tanh(%14)
  %16 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %1, %5)
  %17 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %0)
  %18 : Float(1, 1024, 3072, strides=[3145728, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%17, %16)
  %19 : Float(1024, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%18, %8)
  return (%19)
"""), ("autogen-25", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(16, 128, 1, strides=[128, 1, 1], requires_grad=0, device=cuda:0),
      %4 : Double(requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int,
      %7 : int):
  %8 : int[] = prim::Constant[value=[2048, 768]]()
  %9 : NoneType = prim::Constant()
  %10 : bool = prim::Constant[value=1]()
  %11 : int[] = prim::Constant[value=[-1]]()
  %12 : Float(16, 128, 1, strides=[128, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%3, %4, %7)
  %13 : Float(16, 128, 1, strides=[128, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%2, %11, %10, %9)
  %14 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %13, %6)
  %15 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%1, %14)
  %16 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::div(%15, %12)
  %17 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %0, %5)
  %18 : Float(2048, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%17, %8)
  return (%18)
"""), ("autogen-26", """graph(%0 : Float(1, 8, 2048, 2048, strides=[8, 1, 16384, 8], requires_grad=0, device=cuda:0),
      %1 : Float(8, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : bool = prim::Constant[value=0]()
  %4 : int = prim::Constant[value=-1]()
  %5 : int[] = prim::Constant[value=[8, 2048, 2048]]()
  %6 : int[] = prim::Constant[value=[1, 8, 2048, 2048]]()
  %7 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %6)
  %8 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::add(%7, %0, %2)
  %9 : Float(8, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %5)
  %10 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %6)
  %11 : Float(1, 8, 2048, 2048, strides=[33554432, 4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%10, %4, %3)
  return (%11)
"""), ("autogen-27", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : float = prim::Constant[value=9.9999999999999998e-13]()
  %8 : int[] = prim::Constant[value=[768]]()
  %9 : int[] = prim::Constant[value=[512, 768]]()
  %10 : int[] = prim::Constant[value=[1, 512, 768]]()
  %11 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %10)
  %12 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %3, %6)
  %13 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %9)
  %14 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%14, %2, %5)
  %16 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0), %17 : Tensor, %18 : Tensor = aten::native_layer_norm(%15, %8, %0, %1, %7)
  %19 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %9)
  return (%19, %16)
"""), ("autogen-28", """graph(%0 : Float(128, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(128, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0),
      %3 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0),
      %4 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[512, 128]]()
  %8 : float = prim::Constant[value=9.9999999999999998e-13]()
  %9 : int[] = prim::Constant[value=[128]]()
  %10 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%3, %4, %6)
  %11 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%10, %2, %5)
  %12 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0), %13 : Tensor, %14 : Tensor = aten::native_layer_norm(%11, %9, %0, %1, %8)
  %15 : Float(512, 128, strides=[128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %7)
  return (%15)
"""), ("autogen-29", """graph(%0 : Float(720, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0),
      %1 : Float(720, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 12, 60, 64, 64, 1, strides=[64, 245760, 4096, 64, 1, 64], requires_grad=0, device=cuda:0),
      %3 : Float(1, 12, 60, 64, 64, 1, strides=[64, 245760, 4096, 64, 1, 64], requires_grad=0, device=cuda:0),
      %4 : int,
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[720, 64, 64]]()
  %8 : int[] = prim::Constant[value=[1, 12, 60, 64, 64]]()
  %9 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %8)
  %10 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%2, %8)
  %11 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %8)
  %12 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %8)
  %13 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::add(%12, %11, %6)
  %14 : Float(720, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %7)
  %15 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%14, %8)
  %16 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %10, %5)
  %17 : Float(720, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %7)
  %18 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%17, %8)
  %19 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::add(%18, %9, %4)
  %20 : Float(720, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%19, %7)
  %21 : Float(1, 12, 60, 64, 64, strides=[2949120, 245760, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%20, %8)
  return (%21)
"""), ("autogen-30", """graph(%0 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0),
      %3 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0),
      %4 : int):
  %5 : int[] = prim::Constant[value=[4096, 256]]()
  %6 : float = prim::Constant[value=9.9999999999999998e-13]()
  %7 : int[] = prim::Constant[value=[256]]()
  %8 : int[] = prim::Constant[value=[1, 4096, 256]]()
  %9 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %8)
  %10 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %9, %4)
  %11 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0), %12 : Tensor, %13 : Tensor = aten::native_layer_norm(%10, %7, %0, %1, %6)
  %14 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %5)
  return (%14)
"""), ("autogen-31", """graph(%0 : Float(1, 64, 64, 256, strides=[1048576, 16384, 256, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[4096, 256]]()
  %2 : int[] = prim::Constant[value=[1, 4096, 256]]()
  %3 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %4 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %2)
  %5 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %1)
  return (%5)
"""), ("autogen-32", """graph(%0 : Float(1, 12, 64, 64, 64, strides=[3145728, 262144, 4096, 64, 1], requires_grad=0, device=cuda:0),
      %1 : Float(1, 4096, strides=[4096, 1], requires_grad=0, device=cuda:0)):
  %2 : int[] = prim::Constant[value=[1, 12, 4096, 64]]()
  %3 : int[] = prim::Constant[value=[1, 1, 4096, 1]]()
  %4 : Float(1, 1, 4096, 1, strides=[4096, 4096, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %3)
  %5 : Float(1, 12, 4096, 64, strides=[3145728, 262144, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %6 : Float(1, 12, 4096, 64, strides=[3145728, 262144, 64, 1], requires_grad=0, device=cuda:0) = aten::mul(%5, %4)
  return (%6, %4)
"""), ("autogen-33", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(12, 64, 4096, strides=[262144, 4096, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[12, 64, 4096]]()
  %8 : bool = prim::Constant[value=0]()
  %9 : int = prim::Constant[value=-1]()
  %10 : int[] = prim::Constant[value=[1, 12, 64, 4096]]()
  %11 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %4, %6)
  %12 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::mul(%11, %2)
  %13 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %10)
  %14 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %0)
  %15 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::add(%14, %12, %5)
  %16 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%15, %9, %8)
  %17 : Float(12, 64, 4096, strides=[262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %7)
  return (%17)
"""), ("autogen-34", """graph(%0 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0),
      %3 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : float = prim::Constant[value=9.9999999999999995e-07]()
  %8 : int[] = prim::Constant[value=[384]]()
  %9 : int[] = prim::Constant[value=[1576, 384]]()
  %10 : int[] = prim::Constant[value=[8, 197, 384]]()
  %11 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %10)
  %12 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %3, %6)
  %13 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %9)
  %14 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %14, %5)
  %16 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0), %17 : Tensor, %18 : Tensor = aten::native_layer_norm(%15, %8, %0, %1, %7)
  %19 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %9)
  return (%19, %15)
"""), ("autogen-35", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0),
      %4 : Float(2048, 512, strides=[512, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int,
      %7 : int):
  %8 : int[] = prim::Constant[value=[2048, 512]]()
  %9 : NoneType = prim::Constant()
  %10 : bool = prim::Constant[value=1]()
  %11 : int[] = prim::Constant[value=[-1]]()
  %12 : int[] = prim::Constant[value=[1, 2048, 512]]()
  %13 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %12)
  %14 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%3, %13, %7)
  %15 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::pow(%14, %6)
  %16 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%15, %11, %10, %9)
  %17 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %2, %5)
  %18 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::rsqrt(%17)
  %19 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %18)
  %20 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%1, %19)
  %21 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%20, %0)
  %22 : Float(2048, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%21, %8)
  return (%22)
"""), ("autogen-36", """graph(%0 : Float(32, 512, 28, 28, strides=[401408, 784, 28, 1], requires_grad=0, device=cuda:0),
      %1 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(512, strides=[1], requires_grad=0, device=cuda:0)):
  %5 : bool = prim::Constant[value=1]()
  %6 : int[] = prim::Constant[value=[2, 3]]()
  %7 : NoneType = prim::Constant()
  %8 : int[] = prim::Constant[value=[1]]()
  %9 : int[] = prim::Constant[value=[32, 2, 256, 28, 28]]()
  %10 : float = prim::Constant[value=1.0000000000000001e-05]()
  %11 : float = prim::Constant[value=0.10000000000000001]()
  %12 : bool = prim::Constant[value=0]()
  %13 : Float(32, 512, 28, 28, strides=[401408, 784, 28, 1], requires_grad=0, device=cuda:0), %14 : Tensor, %15 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %12, %11, %10)
  %16 : Float(32, 512, 28, 28, strides=[401408, 784, 28, 1], requires_grad=0, device=cuda:0) = aten::relu(%13)
  %17 : Float(32, 2, 256, 28, 28, strides=[401408, 200704, 784, 28, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %9)
  %18 : Float(32, 256, 28, 28, strides=[200704, 784, 28, 1], requires_grad=0, device=cuda:0) = aten::sum(%17, %8, %12, %7)
  %19 : Float(32, 256, 1, 1, strides=[256, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%18, %6, %5, %7)
  return (%19, %17)
"""), ("autogen-37", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(720, 64, 192, strides=[12288, 192, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[1, 12, 60, 64, 192]]()
  %8 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %4, %6)
  %9 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%8, %2)
  %10 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %7)
  %11 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %0)
  %12 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %9, %5)
  return (%12)
"""), ("autogen-38", """graph(%0 : Float(1, 4096, 256, strides=[2097152, 512, 1], requires_grad=0, device=cuda:0),
      %1 : Float(256, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(256, strides=[1], requires_grad=0, device=cuda:0)):
  %3 : int[] = prim::Constant[value=[4096, 256]]()
  %4 : float = prim::Constant[value=9.9999999999999998e-13]()
  %5 : int[] = prim::Constant[value=[256]]()
  %6 : Float(1, 4096, 256, strides=[1048576, 256, 1], requires_grad=0, device=cuda:0), %7 : Tensor, %8 : Tensor = aten::native_layer_norm(%0, %5, %1, %2, %4)
  %9 : Float(4096, 256, strides=[256, 1], requires_grad=0, device=cuda:0) = aten::reshape(%6, %3)
  return (%9)
"""), ("autogen-39", """graph(%0 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0),
      %1 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0),
      %2 : int,
      %3 : int,
      %4 : int):
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=1]()
  %7 : int[] = prim::Constant[value=[-1]]()
  %8 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%0, %1, %4)
  %9 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::exp(%8)
  %10 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::sum(%9, %7, %6, %5)
  %11 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::log(%10)
  %12 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %1, %3)
  %13 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%0, %12, %2)
  %14 : Float(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::exp(%13)
  return (%14)
"""), ("autogen-40", """graph(%0 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %6 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %10 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %11 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %12 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %13 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %14 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %15 : int,
      %16 : int):
  %17 : float = prim::Constant[value=0.001]()
  %18 : float = prim::Constant[value=0.01]()
  %19 : bool = prim::Constant[value=0]()
  %20 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %21 : Tensor, %22 : Tensor = aten::native_batch_norm(%10, %11, %12, %13, %14, %19, %18, %17)
  %23 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %24 : Tensor, %25 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %19, %18, %17)
  %26 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%23, %20, %16)
  %27 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %28 : Tensor, %29 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %19, %18, %17)
  %30 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%27, %26, %15)
  return (%30)
"""), ("autogen-41", """graph(%0 : Float(12, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[1, 12, 1, 64, 64]]()
  %2 : int[] = prim::Constant[value=[12, 64, 64]]()
  %3 : int = prim::Constant[value=2]()
  %4 : int[] = prim::Constant[value=[1, 12, 64, 64]]()
  %5 : Float(1, 12, 64, 64, strides=[49152, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %4)
  %6 : Float(1, 12, 1, 64, 64, strides=[49152, 4096, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::unsqueeze(%5, %3)
  %7 : Float(1, 12, 64, 64, strides=[49152, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%6, %4)
  %8 : Float(12, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %2)
  %9 : Float(1, 12, 64, 64, strides=[49152, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %4)
  %10 : Float(1, 12, 1, 64, 64, strides=[49152, 4096, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %1)
  return (%10)
"""), ("autogen-42", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(3072, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(512, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0),
      %6 : int,
      %7 : int,
      %8 : float,
      %9 : int):
  %10 : int[] = prim::Constant[value=[512, 3072]]()
  %11 : int[] = prim::Constant[value=[1, 512, 3072]]()
  %12 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %11)
  %13 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%12, %4, %9)
  %14 : Float(512, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%14, %11)
  %16 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::pow(%15, %8)
  %17 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%16, %3)
  %18 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %17, %7)
  %19 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %2)
  %20 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::tanh(%19)
  %21 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%20, %1, %6)
  %22 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%15, %0)
  %23 : Float(1, 512, 3072, strides=[1572864, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%22, %21)
  %24 : Float(512, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%23, %10)
  return (%24)
"""), ("autogen-43", """graph(%0 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %6 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %10 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %11 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %12 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %13 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %14 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %15 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0),
      %16 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %17 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %18 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %19 : Float(80, strides=[1], requires_grad=0, device=cuda:0),
      %20 : int,
      %21 : int,
      %22 : int):
  %23 : float = prim::Constant[value=0.001]()
  %24 : float = prim::Constant[value=0.01]()
  %25 : bool = prim::Constant[value=0]()
  %26 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %27 : Tensor, %28 : Tensor = aten::native_batch_norm(%15, %16, %17, %18, %19, %25, %24, %23)
  %29 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %30 : Tensor, %31 : Tensor = aten::native_batch_norm(%10, %11, %12, %13, %14, %25, %24, %23)
  %32 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%29, %26, %22)
  %33 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %34 : Tensor, %35 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %25, %24, %23)
  %36 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%33, %32, %21)
  %37 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0), %38 : Tensor, %39 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %25, %24, %23)
  %40 : Float(32, 80, 14, 14, strides=[15680, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%37, %36, %20)
  return (%40)
"""), ("autogen-44", """graph(%0 : Float(128, 1024, 7, 7, strides=[50176, 49, 7, 1], requires_grad=0, device=cuda:0),
      %1 : Float(1024, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1024, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(1024, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(1024, strides=[1], requires_grad=0, device=cuda:0)):
  %5 : NoneType = prim::Constant()
  %6 : int[] = prim::Constant[value=[2, 3]]()
  %7 : float = prim::Constant[value=1.0000000000000001e-05]()
  %8 : float = prim::Constant[value=0.10000000000000001]()
  %9 : bool = prim::Constant[value=0]()
  %10 : Float(128, 1024, 7, 7, strides=[50176, 49, 7, 1], requires_grad=0, device=cuda:0), %11 : Tensor, %12 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %9, %8, %7)
  %13 : Float(128, 1024, 7, 7, strides=[50176, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::relu(%10)
  %14 : Float(128, 1024, strides=[1024, 1], requires_grad=0, device=cuda:0) = aten::mean(%13, %6, %9, %5)
  return (%14)
"""), ("autogen-45", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Double(requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %8 : int,
      %9 : int,
      %10 : int):
  %11 : float = prim::Constant[value=9.9999999999999998e-13]()
  %12 : int[] = prim::Constant[value=[768]]()
  %13 : int[] = prim::Constant[value=[4096, 768]]()
  %14 : int[] = prim::Constant[value=[1, 4096, 768]]()
  %15 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %14)
  %16 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %6, %10)
  %17 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%16, %13)
  %18 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%17, %14)
  %19 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %5)
  %20 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%19, %18)
  %21 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%20, %3, %9)
  %22 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %4)
  %23 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%22, %21)
  %24 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::tanh(%23)
  %25 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%24, %3, %8)
  %26 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %2)
  %27 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0) = aten::mul(%26, %25)
  %28 : Float(1, 4096, 768, strides=[3145728, 768, 1], requires_grad=0, device=cuda:0), %29 : Tensor, %30 : Tensor = aten::native_layer_norm(%27, %12, %0, %1, %11)
  %31 : Float(4096, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%28, %13)
  return (%31)
"""), ("autogen-46", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(3072, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(4096, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0),
      %6 : int,
      %7 : int,
      %8 : int):
  %9 : int[] = prim::Constant[value=[4096, 3072]]()
  %10 : int[] = prim::Constant[value=[1, 4096, 3072]]()
  %11 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %10)
  %12 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %4, %8)
  %13 : Float(4096, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %9)
  %14 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %3)
  %16 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%15, %14)
  %17 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %1, %7)
  %18 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %2)
  %19 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %17)
  %20 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::tanh(%19)
  %21 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::add(%20, %1, %6)
  %22 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %0)
  %23 : Float(1, 4096, 3072, strides=[12582912, 3072, 1], requires_grad=0, device=cuda:0) = aten::mul(%22, %21)
  %24 : Float(4096, 3072, strides=[3072, 1], requires_grad=0, device=cuda:0) = aten::reshape(%23, %9)
  return (%24)
"""), ("autogen-47", """graph(%0 : Float(1, 12, 4096, 64, strides=[3145728, 64, 768, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[768, 64, 64]]()
  %2 : int[] = prim::Constant[value=[1, 12, 64, 64, 64]]()
  %3 : Float(1, 12, 64, 64, 64, strides=[768, 64, 49152, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %4 : Float(768, 64, 64, strides=[4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %1)
  return (%4, %3)
"""), ("autogen-48", """graph(%0 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0),
      %3 : Float(2048, 512, strides=[512, 1], requires_grad=0, device=cuda:0),
      %4 : int,
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[2048, 512]]()
  %8 : NoneType = prim::Constant()
  %9 : bool = prim::Constant[value=1]()
  %10 : int[] = prim::Constant[value=[-1]]()
  %11 : int[] = prim::Constant[value=[1, 2048, 512]]()
  %12 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %11)
  %13 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %12, %6)
  %14 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::pow(%13, %5)
  %15 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%14, %10, %9, %8)
  %16 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %1, %4)
  %17 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::rsqrt(%16)
  %18 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %17)
  %19 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%0, %18)
  %20 : Float(2048, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%19, %7)
  return (%20, %13)
"""), ("autogen-49", """graph(%0 : Long(requires_grad=0, device=cuda:0),
      %1 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %2 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %3 : int,
      %4 : int):
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=0]()
  %7 : int[] = prim::Constant[value=[1]]()
  %8 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%1, %2, %4)
  %9 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::div(%8, %0)
  %10 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::pow(%9, %3)
  %11 : Float(96, 128, 128, strides=[16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mean(%10, %7, %6, %5)
  %12 : Float(96, 128, strides=[128, 1], requires_grad=0, device=cuda:0) = aten::mean(%11, %7, %6, %5)
  %13 : Float(96, strides=[1], requires_grad=0, device=cuda:0) = aten::mean(%12, %7, %6, %5)
  return (%13)
"""), ("autogen-50", """graph(%0 : Float(1, 12, 1, 4096, 64, 1, strides=[64, 262144, 64, 64, 1, 64], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[1, 12, 1, 4096, 64]]()
  %2 : Float(1, 12, 1, 4096, 64, strides=[3145728, 262144, 262144, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %1)
  %3 : Float(1, 12, 1, 4096, 64, strides=[3145728, 262144, 262144, 64, 1], requires_grad=0, device=cuda:0) = aten::neg(%2)
  return (%3, %2)
"""), ("autogen-51", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(12, 512, 512, strides=[262144, 512, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 512, strides=[512, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : bool = prim::Constant[value=0]()
  %8 : int = prim::Constant[value=-1]()
  %9 : int[] = prim::Constant[value=[1, 12, 512, 512]]()
  %10 : int[] = prim::Constant[value=[1, 1, 1, 512]]()
  %11 : int[] = prim::Constant[value=[1, 1, 512]]()
  %12 : Float(1, 1, 512, strides=[512, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %11)
  %13 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %10)
  %14 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %13, %6)
  %15 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %2)
  %16 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %9)
  %17 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::div(%16, %0)
  %18 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%17, %15, %5)
  %19 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%18, %8, %7)
  return (%19, %15)
"""), ("autogen-52", """graph(%0 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0),
      %1 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0),
      %6 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %10 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0),
      %11 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %12 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %13 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %14 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %15 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0),
      %16 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %17 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %18 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %19 : Float(64, strides=[1], requires_grad=0, device=cuda:0),
      %20 : int,
      %21 : int,
      %22 : int):
  %23 : float = prim::Constant[value=1.0000000000000001e-05]()
  %24 : float = prim::Constant[value=0.10000000000000001]()
  %25 : bool = prim::Constant[value=0]()
  %26 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0), %27 : Tensor, %28 : Tensor = aten::native_batch_norm(%15, %16, %17, %18, %19, %25, %24, %23)
  %29 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0), %30 : Tensor, %31 : Tensor = aten::native_batch_norm(%10, %11, %12, %13, %14, %25, %24, %23)
  %32 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0), %33 : Tensor, %34 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %25, %24, %23)
  %35 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0), %36 : Tensor, %37 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %25, %24, %23)
  %38 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%35, %32, %22)
  %39 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%38, %29, %21)
  %40 : Float(96, 64, 14, 14, strides=[12544, 196, 14, 1], requires_grad=0, device=cuda:0) = aten::add(%39, %26, %20)
  return (%40)
"""), ("autogen-53", """graph(%0 : Float(128, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(128, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Double(requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : Float(128, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(512, 128, strides=[128, 1], requires_grad=0, device=cuda:0),
      %8 : int,
      %9 : int,
      %10 : float,
      %11 : int):
  %12 : float = prim::Constant[value=1.0000000000000001e-05]()
  %13 : int[] = prim::Constant[value=[128]]()
  %14 : int[] = prim::Constant[value=[512, 128]]()
  %15 : int[] = prim::Constant[value=[1, 512, 128]]()
  %16 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %15)
  %17 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %6, %11)
  %18 : Float(512, 128, strides=[128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%17, %14)
  %19 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%18, %15)
  %20 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::pow(%19, %10)
  %21 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%20, %5)
  %22 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%19, %21, %9)
  %23 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%22, %4)
  %24 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::tanh(%23)
  %25 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%24, %3, %8)
  %26 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%19, %2)
  %27 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%26, %25)
  %28 : Float(1, 512, 128, strides=[65536, 128, 1], requires_grad=0, device=cuda:0), %29 : Tensor, %30 : Tensor = aten::native_layer_norm(%27, %13, %0, %1, %12)
  %31 : Float(512, 128, strides=[128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%28, %14)
  return (%31)
"""), ("autogen-54", """graph(%0 : Float(32, 1000, 13, 13, strides=[169000, 169, 13, 1], requires_grad=0, device=cuda:0)):
  %1 : NoneType = prim::Constant()
  %2 : bool = prim::Constant[value=1]()
  %3 : int[] = prim::Constant[value=[-1, -2]]()
  %4 : Float(32, 1000, 13, 13, strides=[169000, 169, 13, 1], requires_grad=0, device=cuda:0) = aten::relu(%0)
  %5 : Float(32, 1000, 1, 1, strides=[1000, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%4, %3, %2, %1)
  return (%5)
"""), ("autogen-55", """graph(%0 : Float(96, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Long(requires_grad=0, device=cuda:0),
      %2 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %3 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %4 : Float(96, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : int,
      %7 : int,
      %8 : int,
      %9 : int):
  %10 : NoneType = prim::Constant()
  %11 : bool = prim::Constant[value=0]()
  %12 : int[] = prim::Constant[value=[1]]()
  %13 : Float(96, strides=[1], requires_grad=0, device=cuda:0) = aten::add(%4, %5, %9)
  %14 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %3, %8)
  %15 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::div(%14, %1)
  %16 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::pow(%15, %7)
  %17 : Float(96, 128, 128, strides=[16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mean(%16, %12, %11, %10)
  %18 : Float(96, 128, strides=[128, 1], requires_grad=0, device=cuda:0) = aten::mean(%17, %12, %11, %10)
  %19 : Float(96, strides=[1], requires_grad=0, device=cuda:0) = aten::mean(%18, %12, %11, %10)
  %20 : Float(96, strides=[1], requires_grad=0, device=cuda:0) = aten::sub(%0, %19, %6)
  %21 : Float(96, strides=[1], requires_grad=0, device=cuda:0) = aten::div(%20, %13)
  return (%21)
"""), ("autogen-56", """graph(%0 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0),
      %3 : Float(384, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : float = prim::Constant[value=9.9999999999999995e-07]()
  %8 : int[] = prim::Constant[value=[384]]()
  %9 : int[] = prim::Constant[value=[1576, 384]]()
  %10 : int[] = prim::Constant[value=[8, 197, 384]]()
  %11 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %10)
  %12 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %3, %6)
  %13 : Float(1576, 384, strides=[384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %9)
  %14 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::reshape(%13, %10)
  %15 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %14, %5)
  %16 : Float(8, 197, 384, strides=[75648, 384, 1], requires_grad=0, device=cuda:0), %17 : Tensor, %18 : Tensor = aten::native_layer_norm(%15, %8, %0, %1, %7)
  return (%16, %17, %18)
"""), ("autogen-57", """graph(%0 : Float(32, 960, 7, 7, strides=[47040, 49, 7, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[32, 960]]()
  %2 : NoneType = prim::Constant()
  %3 : bool = prim::Constant[value=1]()
  %4 : int[] = prim::Constant[value=[-1, -2]]()
  %5 : Float(32, 960, 1, 1, strides=[960, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%0, %4, %3, %2)
  %6 : Float(32, 960, strides=[960, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %1)
  return (%6)
"""), ("autogen-58", """graph(%0 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %2 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %3 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %4 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %5 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %6 : Double(requires_grad=0, device=cuda:0),
      %7 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %8 : Double(requires_grad=0, device=cuda:0),
      %9 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %10 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %11 : Double(requires_grad=0, device=cuda:0),
      %12 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %13 : Double(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0),
      %14 : Double(requires_grad=0, device=cuda:0),
      %15 : Double(requires_grad=0, device=cuda:0),
      %16 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %17 : Double(requires_grad=0, device=cuda:0),
      %18 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %19 : Double(requires_grad=0, device=cuda:0),
      %20 : Double(requires_grad=0, device=cuda:0),
      %21 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %22 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %23 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %24 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %25 : Double(requires_grad=0, device=cuda:0),
      %26 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %27 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %28 : Double(requires_grad=0, device=cuda:0),
      %29 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %30 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %31 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %32 : Double(requires_grad=0, device=cuda:0),
      %33 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %34 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %35 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %36 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %37 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %38 : Double(requires_grad=0, device=cuda:0),
      %39 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %40 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %41 : Double(requires_grad=0, device=cuda:0),
      %42 : Double(requires_grad=0, device=cuda:0),
      %43 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %44 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %45 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %46 : Double(requires_grad=0, device=cuda:0),
      %47 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %48 : Double(requires_grad=0, device=cuda:0),
      %49 : Double(requires_grad=0, device=cuda:0),
      %50 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %51 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %52 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %53 : Double(requires_grad=0, device=cuda:0),
      %54 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %55 : Double(requires_grad=0, device=cuda:0),
      %56 : Double(requires_grad=0, device=cuda:0),
      %57 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %58 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %59 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %60 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %61 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %62 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %63 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %64 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %65 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %66 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %67 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %68 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %69 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %70 : int,
      %71 : int,
      %72 : int,
      %73 : int,
      %74 : int,
      %75 : int,
      %76 : int,
      %77 : int,
      %78 : int,
      %79 : int,
      %80 : int,
      %81 : int,
      %82 : int,
      %83 : int,
      %84 : int,
      %85 : int,
      %86 : int,
      %87 : int,
      %88 : int,
      %89 : int,
      %90 : int,
      %91 : int,
      %92 : int,
      %93 : int,
      %94 : int,
      %95 : int,
      %96 : int,
      %97 : int,
      %98 : int,
      %99 : int,
      %100 : int,
      %101 : int,
      %102 : int,
      %103 : int,
      %104 : int,
      %105 : int,
      %106 : int,
      %107 : int,
      %108 : int,
      %109 : int,
      %110 : int,
      %111 : int,
      %112 : int,
      %113 : int,
      %114 : int,
      %115 : int,
      %116 : int,
      %117 : int,
      %118 : int,
      %119 : int,
      %120 : int,
      %121 : int,
      %122 : int,
      %123 : int,
      %124 : int):
  %125 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%68, %69, %124)
  %126 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%23, %9)
  %127 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%67, %22)
  %128 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%127, %21)
  %129 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%125, %128, %123)
  %130 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%129, %7)
  %131 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%66, %21)
  %132 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%65, %22)
  %133 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%132, %21)
  %134 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%64, %29)
  %135 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%63, %30)
  %136 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%61, %7)
  %137 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%136, %22)
  %138 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%137, %21)
  %139 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%61, %62)
  %140 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%60, %11)
  %141 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%140, %58)
  %142 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%59, %11)
  %143 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%142, %58)
  %144 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%143, %141, %122)
  %145 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%144, %139, %121)
  %146 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%145, %138, %120)
  %147 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%146, %135, %119)
  %148 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%147, %134, %118)
  %149 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%148, %3)
  %150 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%57, %11)
  %151 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%150, %56, %117)
  %152 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %151)
  %153 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%54, %55, %116)
  %154 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%153, %152, %115)
  %155 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %154)
  %156 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%52, %53, %114)
  %157 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%156, %155, %113)
  %158 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%51, %157)
  %159 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%50, %11)
  %160 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%159, %49, %112)
  %161 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %160)
  %162 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%47, %48, %111)
  %163 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%162, %161, %110)
  %164 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %163)
  %165 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%45, %46, %109)
  %166 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%165, %164, %108)
  %167 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%166, %158, %107)
  %168 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%44, %167)
  %169 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%43, %11)
  %170 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%169, %42, %106)
  %171 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %170)
  %172 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%40, %41, %105)
  %173 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%172, %171, %104)
  %174 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%39, %173)
  %175 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%37, %38, %103)
  %176 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%175, %174, %102)
  %177 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%176, %168, %101)
  %178 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%177, %149, %100)
  %179 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%178, %133, %99)
  %180 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%179, %36)
  %181 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%180, %131, %98)
  %182 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%181, %130, %97)
  %183 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%182, %126, %96)
  %184 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%183, %20)
  %185 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%184, %3)
  %186 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%185, %35)
  %187 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::reciprocal(%30)
  %188 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%187, %28)
  %189 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%34, %28)
  %190 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%33, %28)
  %191 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%190, %32, %95)
  %192 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%191, %189, %94)
  %193 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%192, %31, %93)
  %194 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %193)
  %195 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%194, %188)
  %196 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%195, %28, %92)
  %197 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::pow(%30, %91)
  %198 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%194, %197)
  %199 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%198, %29)
  %200 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%27, %28)
  %201 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%26, %11)
  %202 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%201, %25, %90)
  %203 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%202, %200, %89)
  %204 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%203, %24, %88)
  %205 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %204)
  %206 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%205, %188)
  %207 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%206, %199, %87)
  %208 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%23, %20)
  %209 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%208, %3)
  %210 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%209, %207)
  %211 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::div(%210, %196)
  %212 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::log(%196)
  %213 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%208, %22)
  %214 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%213, %212)
  %215 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%214, %21)
  %216 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%129, %20)
  %217 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%216, %3)
  %218 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%217, %212)
  %219 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %19)
  %220 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%219, %13)
  %221 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%18, %11)
  %222 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%221, %3)
  %223 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %17, %86)
  %224 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%223, %222, %85)
  %225 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%224, %220, %84)
  %226 : Double(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %15)
  %227 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%226, %225)
  %228 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%227, %12)
  %229 : Double(1, 1, 26, strides=[26, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %14)
  %230 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%221, %12)
  %231 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %11)
  %232 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%231, %9)
  %233 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%232, %3)
  %234 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%7, %8)
  %235 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%234, %3)
  %236 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%5, %6, %83)
  %237 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%236, %4, %82)
  %238 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%237, %235, %81)
  %239 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%238, %233, %80)
  %240 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%239, %230, %79)
  %241 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%240, %229, %78)
  %242 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%226, %241)
  %243 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%242, %3)
  %244 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%243, %228, %77)
  %245 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%244, %218, %76)
  %246 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%245, %215, %75)
  %247 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%246, %211, %74)
  %248 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%247, %186, %73)
  %249 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%248, %2, %72)
  %250 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::sub(%249, %1, %71)
  %251 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%250, %0, %70)
  return (%251)
"""), ("autogen-59", """graph(%0 : Long(1, 12, 4096, strides=[49152, 4096, 1], requires_grad=0, device=cuda:0),
      %1 : Long(requires_grad=0, device=cuda:0),
      %2 : Long(1, 12, 1, 4096, strides=[49152, 4096, 4096, 1], requires_grad=0, device=cuda:0),
      %3 : Long(1, 12, 1, 1, strides=[1, 0, 1, 1], requires_grad=0, device=cuda:0),
      %4 : int,
      %5 : int):
  %6 : int[] = prim::Constant[value=[1, 12, 4096]]()
  %7 : Long(1, 12, 1, 4096, strides=[49152, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %3, %5)
  %8 : Long(1, 12, 4096, strides=[49152, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %6)
  %9 : Long(1, 12, 4096, strides=[49152, 4096, 1], requires_grad=0, device=cuda:0) = aten::mul(%8, %1)
  %10 : Long(1, 12, 4096, strides=[49152, 4096, 1], requires_grad=0, device=cuda:0) = aten::add(%9, %0, %4)
  return (%10)
"""), ("autogen-60", """graph(%0 : Float(requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Float(1, 12, 4096, 64, strides=[3145728, 262144, 64, 1], requires_grad=0, device=cuda:0),
      %3 : int,
      %4 : int):
  %5 : NoneType = prim::Constant()
  %6 : bool = prim::Constant[value=1]()
  %7 : int[] = prim::Constant[value=[-1]]()
  %8 : int[] = prim::Constant[value=[1, 12, 64, 64, 64]]()
  %9 : Float(1, 12, 64, 64, 64, strides=[3145728, 262144, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::reshape(%2, %8)
  %10 : Float(1, 12, 64, 64, 64, strides=[3145728, 262144, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::pow(%9, %4)
  %11 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%10, %7, %6, %5)
  %12 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %1, %3)
  %13 : Float(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::rsqrt(%12)
  %14 : Float(1, 12, 64, 64, 64, strides=[3145728, 262144, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::mul(%9, %13)
  %15 : Float(1, 12, 64, 64, 64, strides=[3145728, 262144, 4096, 64, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %0)
  return (%15, %9)
"""), ("autogen-61", """graph(%0 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(2048, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %3 : int,
      %4 : int):
  %5 : int[] = prim::Constant[value=[2048, 768]]()
  %6 : int[] = prim::Constant[value=[16, 128, 768]]()
  %7 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%2, %6)
  %8 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%7, %1, %4)
  %9 : Float(2048, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%8, %5)
  %10 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %6)
  %11 : Float(16, 128, 768, strides=[98304, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%0, %10, %3)
  %12 : Float(2048, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %5)
  return (%12, %11)
"""), ("autogen-62", """graph(%0 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0),
      %1 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %3 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %5 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0),
      %6 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %7 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %8 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %9 : Float(2048, strides=[1], requires_grad=0, device=cuda:0),
      %10 : int):
  %11 : int[] = prim::Constant[value=[32, 2048]]()
  %12 : NoneType = prim::Constant()
  %13 : bool = prim::Constant[value=1]()
  %14 : int[] = prim::Constant[value=[-1, -2]]()
  %15 : float = prim::Constant[value=1.0000000000000001e-05]()
  %16 : float = prim::Constant[value=0.10000000000000001]()
  %17 : bool = prim::Constant[value=0]()
  %18 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0), %19 : Tensor, %20 : Tensor = aten::native_batch_norm(%5, %6, %7, %8, %9, %17, %16, %15)
  %21 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0), %22 : Tensor, %23 : Tensor = aten::native_batch_norm(%0, %1, %2, %3, %4, %17, %16, %15)
  %24 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::add(%21, %18, %10)
  %25 : Float(32, 2048, 7, 7, strides=[100352, 49, 7, 1], requires_grad=0, device=cuda:0) = aten::relu(%24)
  %26 : Float(32, 2048, 1, 1, strides=[2048, 1, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%25, %14, %13, %12)
  %27 : Float(32, 2048, strides=[2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%26, %11)
  return (%27)
"""), ("autogen-63", """graph(%0 : Float(480, 1, 1, 3, strides=[13, 3, 3, 1], requires_grad=0, device=cuda:0),
      %1 : Long(requires_grad=0, device=cuda:0),
      %2 : Float(480, 1, 64, 2, 64, 2, strides=[16384, 16384, 64, 8192, 1, 4096], requires_grad=0, device=cuda:0),
      %3 : int,
      %4 : int):
  %5 : int[] = prim::Constant[value=[480, 128, 128, 1]]()
  %6 : int[] = prim::Constant[value=[480, 128, 128]]()
  %7 : int[] = prim::Constant[value=[480, 1, 128, 128]]()
  %8 : Float(480, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%2, %7)
  %9 : Float(480, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sigmoid(%8)
  %10 : Float(480, 128, 128, strides=[16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %6)
  %11 : Float(480, 128, 128, strides=[16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%1, %10, %4)
  %12 : Float(480, 128, 128, strides=[16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%1, %11, %3)
  %13 : Float(480, 128, 128, 1, strides=[16384, 128, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%12, %5)
  %14 : Float(480, 128, 128, 3, strides=[49152, 384, 3, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %0)
  return (%14, %13)
"""), ("autogen-64", """graph(%0 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %1 : Double(requires_grad=0, device=cuda:0),
      %2 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %5 : Double(requires_grad=0, device=cuda:0),
      %6 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0),
      %7 : Double(requires_grad=0, device=cuda:0),
      %8 : int,
      %9 : int,
      %10 : int,
      %11 : int):
  %12 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%6, %7)
  %13 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %5)
  %14 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%13, %3, %11)
  %15 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::mul(%2, %14)
  %16 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%0, %1, %10)
  %17 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %15, %9)
  %18 : Double(204, 204, 26, strides=[5304, 26, 1], requires_grad=0, device=cuda:0) = aten::add(%17, %12, %8)
  return (%18)
"""), ("autogen-65", """graph(%0 : Float(20005, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(2048, 20005, strides=[20005, 1], requires_grad=0, device=cuda:0),
      %2 : int):
  %3 : int[] = prim::Constant[value=[2048, 20005]]()
  %4 : int[] = prim::Constant[value=[16, 128, 20005]]()
  %5 : Float(16, 128, 20005, strides=[2560640, 20005, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %4)
  %6 : Float(16, 128, 20005, strides=[2560640, 20005, 1], requires_grad=0, device=cuda:0) = aten::add(%5, %0, %2)
  %7 : Float(2048, 20005, strides=[20005, 1], requires_grad=0, device=cuda:0) = aten::reshape(%6, %3)
  %8 : Float(16, 128, 20005, strides=[2560640, 20005, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %4)
  return (%8)
"""), ("autogen-66", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 1024, 768, strides=[786432, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(1024, 768, strides=[768, 1], requires_grad=0, device=cuda:0),
      %4 : int):
  %5 : int[] = prim::Constant[value=[1024, 768]]()
  %6 : float = prim::Constant[value=1.0000000000000001e-05]()
  %7 : int[] = prim::Constant[value=[768]]()
  %8 : int[] = prim::Constant[value=[1, 1024, 768]]()
  %9 : Float(1, 1024, 768, strides=[786432, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %8)
  %10 : Float(1, 1024, 768, strides=[786432, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %9, %4)
  %11 : Float(1, 1024, 768, strides=[786432, 768, 1], requires_grad=0, device=cuda:0), %12 : Tensor, %13 : Tensor = aten::native_layer_norm(%10, %7, %0, %1, %6)
  %14 : Float(1, 1024, 768, strides=[786432, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %8)
  %15 : Float(1024, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%14, %5)
  return (%15)
"""), ("autogen-67", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(720, 64, 192, strides=[12288, 192, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 60, 64, 1, strides=[3840, 64, 1, 1], requires_grad=0, device=cuda:0),
      %5 : Float(1, 60, 1, 192, strides=[11520, 192, 1, 1], requires_grad=0, device=cuda:0),
      %6 : int,
      %7 : int):
  %8 : int[] = prim::Constant[value=[1, 12, 60, 64, 192]]()
  %9 : int = prim::Constant[value=1]()
  %10 : Float(1, 60, 64, 192, strides=[737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%4, %5)
  %11 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::unsqueeze(%10, %9)
  %12 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %11, %7)
  %13 : Float(1, 1, 60, 64, 192, strides=[737280, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%12, %2)
  %14 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %8)
  %15 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%14, %0)
  %16 : Float(1, 12, 60, 64, 192, strides=[8847360, 737280, 12288, 192, 1], requires_grad=0, device=cuda:0) = aten::add(%15, %13, %6)
  return (%16, %11)
"""), ("autogen-68", """graph(%0 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0),
      %3 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %4 : Float(1, 512, 768, 1, 1, strides=[768, 768, 1, 768, 768], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[512, 768]]()
  %8 : float = prim::Constant[value=9.9999999999999998e-13]()
  %9 : int[] = prim::Constant[value=[768]]()
  %10 : int[] = prim::Constant[value=[1, 512, 768]]()
  %11 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %10)
  %12 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%11, %3, %6)
  %13 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0) = aten::add(%2, %12, %5)
  %14 : Float(1, 512, 768, strides=[393216, 768, 1], requires_grad=0, device=cuda:0), %15 : Tensor, %16 : Tensor = aten::native_layer_norm(%13, %9, %0, %1, %8)
  %17 : Float(512, 768, strides=[768, 1], requires_grad=0, device=cuda:0) = aten::reshape(%14, %7)
  return (%17, %14)
"""), ("autogen-69", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(12, 64, 4096, strides=[262144, 4096, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 4096, strides=[4096, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : int[] = prim::Constant[value=[12, 64, 4096]]()
  %8 : bool = prim::Constant[value=0]()
  %9 : int = prim::Constant[value=-1]()
  %10 : int[] = prim::Constant[value=[1, 12, 64, 4096]]()
  %11 : int[] = prim::Constant[value=[1, 1, 1, 4096]]()
  %12 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %11)
  %13 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %12, %6)
  %14 : Float(1, 1, 1, 4096, strides=[4096, 4096, 4096, 1], requires_grad=0, device=cuda:0) = aten::mul(%13, %2)
  %15 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %10)
  %16 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::mul(%15, %0)
  %17 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::add(%16, %14, %5)
  %18 : Float(1, 12, 64, 4096, strides=[3145728, 262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%17, %9, %8)
  %19 : Float(12, 64, 4096, strides=[262144, 4096, 1], requires_grad=0, device=cuda:0) = aten::reshape(%18, %7)
  return (%19, %12)
"""), ("autogen-70", """graph(%0 : Long(1, 12, 64, 64, strides=[49152, 4096, 64, 1], requires_grad=0, device=cuda:0),
      %1 : Long(1, 12, 64, 128, strides=[98304, 8192, 128, 1], requires_grad=0, device=cuda:0)):
  %2 : int[] = prim::Constant[value=[1, 12, 64, 64, 1]]()
  %3 : int[] = prim::Constant[value=[1, 12, 64, 1, 128]]()
  %4 : Long(1, 12, 64, 1, 128, strides=[98304, 8192, 128, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %3)
  %5 : Long(1, 12, 64, 64, 1, strides=[49152, 4096, 64, 1, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %6 : Bool(1, 12, 64, 64, 128, strides=[6291456, 524288, 8192, 128, 1], requires_grad=0, device=cuda:0) = aten::ne(%5, %4)
  return (%6)
"""), ("autogen-71", """graph(%0 : Float(512, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : int,
      %4 : int):
  %5 : int[] = prim::Constant[value=[2048, 512]]()
  %6 : NoneType = prim::Constant()
  %7 : bool = prim::Constant[value=1]()
  %8 : int[] = prim::Constant[value=[-1]]()
  %9 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::pow(%1, %4)
  %10 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::mean(%9, %8, %7, %6)
  %11 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::add(%10, %2, %3)
  %12 : Float(1, 2048, 1, strides=[2048, 1, 1], requires_grad=0, device=cuda:0) = aten::rsqrt(%11)
  %13 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%1, %12)
  %14 : Float(1, 2048, 512, strides=[1048576, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%0, %13)
  %15 : Float(2048, 512, strides=[512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%14, %5)
  return (%15)
"""), ("autogen-72", """graph(%0 : Long(2232, strides=[1], requires_grad=0, device=cuda:0),
      %1 : Long(2232, strides=[1], requires_grad=0, device=cuda:0),
      %2 : Long(requires_grad=0, device=cuda:0),
      %3 : Long(1, 12, 62, 3, strides=[2232, 186, 3, 1], requires_grad=0, device=cuda:0),
      %4 : int,
      %5 : int):
  %6 : int[] = prim::Constant[value=[2232]]()
  %7 : Long(2232, strides=[1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %6)
  %8 : Long(2232, strides=[1], requires_grad=0, device=cuda:0) = aten::mul(%1, %2)
  %9 : Long(2232, strides=[1], requires_grad=0, device=cuda:0) = aten::add(%7, %8, %5)
  %10 : Long(2232, strides=[1], requires_grad=0, device=cuda:0) = aten::add(%7, %0, %4)
  return (%10, %9)
"""), ("autogen-73", """graph(%0 : Long(requires_grad=0, device=cuda:0),
      %1 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %2 : Long(requires_grad=0, device=cuda:0),
      %3 : Float(96, 1, 1, 128, 128, strides=[81920, 16384, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %4 : Float(96, 1, 3, 128, 128, strides=[245760, 49152, 1, 384, 3], requires_grad=0, device=cuda:0),
      %5 : Float(96, 1, 1, 128, 128, strides=[81920, 16384, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %6 : Float(96, 1, 3, 128, 128, strides=[245760, 49152, 1, 384, 3], requires_grad=0, device=cuda:0),
      %7 : Float(96, 1, 1, 128, 128, strides=[81920, 16384, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %8 : Float(96, 1, 3, 128, 128, strides=[245760, 49152, 1, 384, 3], requires_grad=0, device=cuda:0),
      %9 : Float(96, 1, 1, 128, 128, strides=[81920, 16384, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %10 : Float(96, 1, 3, 128, 128, strides=[245760, 49152, 1, 384, 3], requires_grad=0, device=cuda:0),
      %11 : Float(96, 1, 1, 128, 128, strides=[81920, 16384, 16384, 128, 1], requires_grad=0, device=cuda:0),
      %12 : Float(96, 1, 3, 128, 128, strides=[245760, 49152, 1, 384, 3], requires_grad=0, device=cuda:0),
      %13 : int,
      %14 : int,
      %15 : int,
      %16 : int,
      %17 : int,
      %18 : int,
      %19 : int,
      %20 : int,
      %21 : int,
      %22 : int):
  %23 : int[] = prim::Constant[value=[96, 1, 128, 128]]()
  %24 : int[] = prim::Constant[value=[96, 3, 128, 128]]()
  %25 : Float(96, 3, 128, 128, strides=[245760, 1, 384, 3], requires_grad=0, device=cuda:0) = aten::reshape(%12, %24)
  %26 : Float(96, 1, 128, 128, strides=[81920, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%11, %23)
  %27 : Float(96, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %26, %22)
  %28 : Float(96, 3, 128, 128, strides=[245760, 1, 384, 3], requires_grad=0, device=cuda:0) = aten::reshape(%10, %24)
  %29 : Float(96, 1, 128, 128, strides=[81920, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%9, %23)
  %30 : Float(96, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %29, %21)
  %31 : Float(96, 3, 128, 128, strides=[245760, 1, 384, 3], requires_grad=0, device=cuda:0) = aten::reshape(%8, %24)
  %32 : Float(96, 1, 128, 128, strides=[81920, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%7, %23)
  %33 : Float(96, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %32, %20)
  %34 : Float(96, 3, 128, 128, strides=[245760, 1, 384, 3], requires_grad=0, device=cuda:0) = aten::reshape(%6, %24)
  %35 : Float(96, 1, 128, 128, strides=[81920, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%5, %23)
  %36 : Float(96, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %35, %19)
  %37 : Float(96, 3, 128, 128, strides=[245760, 1, 384, 3], requires_grad=0, device=cuda:0) = aten::reshape(%4, %24)
  %38 : Float(96, 1, 128, 128, strides=[81920, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::reshape(%3, %23)
  %39 : Float(96, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::sub(%2, %38, %18)
  %40 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::div(%1, %0)
  %41 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%40, %39)
  %42 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%41, %37, %17)
  %43 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%42, %36)
  %44 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%43, %34, %16)
  %45 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%44, %33)
  %46 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%45, %31, %15)
  %47 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%46, %30)
  %48 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%47, %28, %14)
  %49 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%48, %27)
  %50 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::add(%49, %25, %13)
  %51 : Float(96, 3, 128, 128, strides=[49152, 16384, 128, 1], requires_grad=0, device=cuda:0) = aten::mul(%50, %0)
  return (%51)
"""), ("autogen-74", """graph(%0 : Long(200, 200, strides=[204, 1], requires_grad=0, device=cuda:0),
      %1 : Long(requires_grad=0, device=cuda:0),
      %2 : int,
      %3 : int):
  %4 : Long(200, 200, strides=[200, 1], requires_grad=0, device=cuda:0) = aten::sub(%0, %1, %3)
  %5 : Bool(200, 200, strides=[200, 1], requires_grad=0, device=cuda:0) = aten::ge(%4, %2)
  return (%5, %4)
"""), ("autogen-75", """graph(%0 : Double(requires_grad=0, device=cuda:0),
      %1 : Float(12, 512, 512, strides=[262144, 512, 1], requires_grad=0, device=cuda:0),
      %2 : Double(requires_grad=0, device=cuda:0),
      %3 : Double(requires_grad=0, device=cuda:0),
      %4 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0),
      %5 : int,
      %6 : int):
  %7 : bool = prim::Constant[value=0]()
  %8 : int = prim::Constant[value=-1]()
  %9 : int[] = prim::Constant[value=[1, 12, 512, 512]]()
  %10 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0) = aten::sub(%3, %4, %6)
  %11 : Float(1, 1, 1, 512, strides=[512, 512, 512, 1], requires_grad=0, device=cuda:0) = aten::mul(%10, %2)
  %12 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::reshape(%1, %9)
  %13 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::div(%12, %0)
  %14 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::add(%13, %11, %5)
  %15 : Float(1, 12, 512, 512, strides=[3145728, 262144, 512, 1], requires_grad=0, device=cuda:0) = aten::_softmax(%14, %8, %7)
  return (%15, %11)
"""), ("autogen-76", """graph(%0 : Float(2048, 2048, strides=[2048, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[2048, 2048]]()
  %2 : int[] = prim::Constant[value=[1, 2048, 2048]]()
  %3 : Float(1, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%0, %2)
  %4 : Float(1, 2048, 2048, strides=[4194304, 2048, 1], requires_grad=0, device=cuda:0) = aten::relu(%3)
  %5 : Float(2048, 2048, strides=[2048, 1], requires_grad=0, device=cuda:0) = aten::reshape(%4, %1)
  return (%5)
""")]
