class Prefetcher:
    def __init__(self, data_loader, device):
        self.data_loader = iter(data_loader)
        self.device = device
        self.images = None
        self.targets = None
        self.loader_stream = torch.cuda.Stream()
        self.done = False

    def __iter__(self):
        return self

    def prefetch(self):
        try:
            with torch.cuda.stream(self.loader_stream):
                self.images, self.targets, _ = next(self.data_loader)
                self.images = self.images.to(self.device)
                self.targets = [target.to(self.device, non_blocking=True) for target in self.targets]
        except StopIteration:
            self.images, self.targets = None, None
            self.done = True

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.loader_stream)
        if self.images is None and not self.done:
            self.prefetch()
        if self.done:
            raise StopIteration()
        else:
            images, targets = self.images, self.targets
            self.images, self.targets = None, None
            return images, targets
