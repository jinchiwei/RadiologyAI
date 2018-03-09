"""
class_activities = {
	0:'activity_standing',
	1:'activity_walking'
}
"""



class Diva2DImageDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		"""
		Args:
				root_dir (string): root directory, data should be organized as
											root/{activity_name}/{sample}/frame_N.png
				transform (callable, optional): Optional transform to be applied
						on a sample.
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.sample_paths = []
		self._init()


	def __len__(self):
		return len(self.sample_paths)

	def __getitem__(self, idx):
		img_path,label = self.sample_paths[idx]
		x = io.imread(img_path)[:,:,:3]
		if self.transform:
			x = self.transform(x)
		return (x,label)

	def _init(self):
		data_root = self.root_dir
		activities = list(activity_classes.keys())
		for act in activities:
			act_dir = os.path.join(data_root, act)
			samples = os.listdir(act_dir)
			for sample in samples:
				sample_dir = os.path.join(act_dir, sample)
				for img_name in os.listdir(sample_dir):
					img_path = os.path.join(sample_dir, img_name)
					self.sample_paths.append((img_path,activity_classes[act]))