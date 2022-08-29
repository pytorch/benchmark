import numpy as np
import os
import sys
from enum import Enum

class RecordKind(Enum):
  MemIntensive = 0
  MathIntensive = 1
  MEMSET = 2
  D2D = 3
  D2H = 4
  H2D = 5
  Unknown = 6

KnownMathOp = ['gemm', 'gemv', 'cutlass', 'conv', 'winograd', "Conv"]
class Record:
  def __init__(self):
    self.kind = RecordKind.Unknown

def is_known_math_kernel(name):
  return (any(math_op_name in name for math_op_name in KnownMathOp))

def parse_one_line(line):
  r = Record()
  items = line.strip().split(',')
  r.duration = int(items[1])
  r.name = ','.join(items[19:])

  if r.name == '[CUDA memcpy DtoH]':
    r.kind = RecordKind.D2H
  elif r.name == '[CUDA memcpy DtoD]':
    r.kind = RecordKind.D2D
  elif r.name == '[CUDA memcpy HtoD]':
    r.kind = RecordKind.H2D
  elif r.name == '[CUDA memset]':
    r.kind = RecordKind.MEMSET
  elif is_known_math_kernel(r.name):
    r.kind = RecordKind.MathIntensive
  else:
    r.kind = RecordKind.MemIntensive
  return r

class NsysResults:
  def __init__(self):
    self.records = []
    self.math_records = []
    self.mem_records = []
    self.d2h_or_h2d_records = []
    self.d2d_or_memset_records = []
    self.total_time_in_ns = 0
    self.total_math_time_in_ns = 0
    self.total_mem_time_in_ns = 0
    self.total_d2h_or_h2d_time_in_ns = 0
    self.total_d2d_or_memset_time_in_ns = 0

  def append(self, r):
    self.records.append(r)
    self.total_time_in_ns += r.duration
    if r.kind == RecordKind.MemIntensive:
      self.total_mem_time_in_ns += r.duration
      self.mem_records.append(r)
    elif r.kind == RecordKind.MathIntensive:
      self.total_math_time_in_ns += r.duration
      self.math_records.append(r)
    elif r.kind == RecordKind.H2D or r.kind == RecordKind.D2H:
      self.total_d2h_or_h2d_time_in_ns += r.duration
      self.d2h_or_h2d_records.append(r)
    elif r.kind == RecordKind.D2D or r.kind == RecordKind.MEMSET:
      self.total_d2d_or_memset_time_in_ns += r.duration
      self.d2d_or_memset_records.append(r)
    else:
      raise RuntimeError("Unknown record kind")

  def report(self):
    cmp_func = lambda x: x.duration
    sorted(self.records, key=cmp_func, reverse=True)
    sorted(self.math_records, key=cmp_func, reverse=True)
    sorted(self.mem_records, key=cmp_func, reverse=True)
    sorted(self.d2h_or_h2d_records, key=cmp_func, reverse=True)
    sorted(self.d2d_or_memset_records, key=cmp_func, reverse=True)

    header_template = '{kind:12s} | {num_calls:5d} calls | {num_unique_calls:5d} unique calls | {total_time:7.2f} us | {percentage:6.2f}%'

    def print_topk_records(records, topk = -1):
        group_by_name = {}
        for r in records:
          group_by_name.setdefault(r.name, [])
          group_by_name[r.name].append(r)
        groups = sorted([ group_by_name[k] for k in group_by_name], key = lambda rs : sum([r.duration for r in rs]), reverse=True)

        if topk == -1 or topk > len(group_by_name):
          topk = len(group_by_name)

        for i in range(topk):
          group_time = [ r.duration for r in groups[i]]
          print('  #{:02d}/{:03d} | {:7.2f} us | {:6.2f} | {:5d} calls | min {:7.2f} us | max {:7.2f} | avg {:7.2f} | median {:7.2f} | {}'.format(
            i, len(groups),
            np.sum(group_time) / 1e3, np.sum(group_time) / self.total_time_in_ns * 100,
            len(group_time), np.min(group_time) / 1e3,
            np.max(group_time) / 1e3, np.mean(group_time) / 1e3, np.median(group_time) / 1e3,
            groups[i][0].name
          ))
        print()


    print(header_template.format(
        kind='All',
        num_calls=len(self.records),
        num_unique_calls=len(set([r.name for r in self.records])),
        total_time=self.total_time_in_ns / 1e3,
        percentage=100.0)
    )
    print('=' * 80)
    print_topk_records(self.records, topk=0)
    
    print(header_template.format(
        kind='Math',
        num_calls=len(self.math_records),
        num_unique_calls=len(set([r.name for r in self.math_records])),
        total_time=self.total_math_time_in_ns / 1e3,
        percentage=self.total_math_time_in_ns / self.total_time_in_ns * 100)
    )
    print('=' * 80)
    print_topk_records(self.math_records, topk=5)

    print(header_template.format(
        kind='Mem',
        num_calls=len(self.mem_records),
        num_unique_calls=len(set([r.name for r in self.mem_records])),
        total_time=self.total_mem_time_in_ns / 1e3,
        percentage=self.total_mem_time_in_ns / self.total_time_in_ns * 100)
    )
    print('=' * 80)
    print_topk_records(self.mem_records, topk=10)

    print(header_template.format(
        kind='D2H/H2D',
        num_calls=len(self.d2h_or_h2d_records),
        num_unique_calls=len(set([r.name for r in self.d2h_or_h2d_records])),
        total_time=self.total_d2h_or_h2d_time_in_ns / 1e3,
        percentage=self.total_d2h_or_h2d_time_in_ns / self.total_time_in_ns * 100)
    )
    print('=' * 80)
    print_topk_records(self.d2h_or_h2d_records, topk=5)

    print(header_template.format(
        kind='D2D/MEMSET',
        num_calls=len(self.d2d_or_memset_records),
        num_unique_calls=len(set([r.name for r in self.d2d_or_memset_records])),
        total_time=self.total_d2d_or_memset_time_in_ns / 1e3,
        percentage=self.total_d2d_or_memset_time_in_ns / self.total_time_in_ns * 100)
    )
    print('=' * 80)
    print_topk_records(self.d2d_or_memset_records, topk=5)

    # print(f'ALL | total {len(self.records)} calls | {len(set([r.name for r in self.records]))} unique calls | ', end='')
    # print('total')

def main(argv):
  if len(argv) != 2:
    print('Usage:\n\tpython parse_nsys_result.py result_file_path\n')
    sys.exit(0)
  
  header = 'Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MBps),SrcMemKd,DstMemKd,Device,Ctx,Strm,Name'
  after_header = False
  records = NsysResults()
  with open(argv[-1]) as fd:
    for l in fd:
      if header in l:
        after_header = True
        continue
      if not after_header:
        continue
      l = l.strip()
      if 'xxxxxx END NSYS xxxxxx' in l:
        break
      if len(l) == 0: continue 
      r = parse_one_line(l)
      if r.kind == RecordKind.Unknown:
        print('[WARNING] Ingnore unkown record line: ', l)
        continue
      records.append(r)
    records.report()

if __name__ == '__main__':
  main(sys.argv)
