--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'
require 'lfs'
local ffi = require 'ffi'

local dataset = torch.class('dataLoader')


function dataset:__init(data_dir, category, loadSize, split, background)
	 self.data_dir = data_dir
	 self.category = category 
	 self.loadSize = loadSize
	 self.frames_inc = 1
	 self.frames_max = 10
	 self.n_frames = 10
	 self.split = split
	 self.background = background
end

-- size()
function dataset:size()
	return {self.metadata.n_samples, self.metadata.n_train, self.metadata.n_test}
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local data1 = torch.Tensor(quantity,
		       self.loadSize[1], self.loadSize[2], self.loadSize[3])
   local data2 = torch.Tensor(quantity,
		       self.loadSize[1], self.loadSize[2], self.loadSize[3])
	-- local maps = torch.Tensor(quantity, 
			--		 1, self.loadSize[2], self.loadSize[3])
   local trans = torch.Tensor(quantity,10)
   for i=1,quantity do
      local out1, out2, map, transform = self:get()
      data1[i]:copy(out1)
      data2[i]:copy(out2)
     -- maps[i][1]:copy(map)
			trans[i]:copy(transform)
   end
   return data1, data2, trans 
end

function dataset:get()
	local model_id
	if self.split == 'train' then
		model_id = self.metadata.train_indices[torch.random(self.metadata.n_train)]
	elseif self.split == 'val' then
		model_id = self.metadata.val_indices[torch.random(self.metadata.n_val)]
	elseif self.split == 'test' then
		model_id = self.metadata.test_indices[torch.random(self.metadata.n_test)]
	end

	local src_theta_id = torch.random(self.n_frames) 
	local src_theta = (src_theta_id-1)*self.theta_inc
	local transform_id = torch.random(10)
	local one_hot = torch.zeros(10)
	one_hot:scatter(1,torch.LongTensor{transform_id},1)

	local dst_frame = scr_frame + transform_id

	--TODO: Fix corner cases.
	--if dst_theta < 0 then
	--	dst_theta = 360 + dst_theta 
	--elseif dst_theta > 350 then
	--	dst_theta = dst_theta - 360
	--end

	--source image and target image
	local model_name = ffi.string(torch.data(self.metadata.models[model_id]))
	-- TODO: add a function to pass the string in proper way. 	
	local imgpath1 = self.data_dir .. '/' .. model_name .. 
			'/02/' .. string.format('00000%d.png',src_theta)
	local imgpath2 = self.data_dir .. self.category .. '/' .. model_name .. 
			'/02/' .. string.format('00000%d.png',dst_theta)
	local im1 = self:sampleHookTrain(imgpath1)
	local im2 = self:sampleHookTrain(imgpath2)
	local im1_rgb, im2_rgb
	--local map = self.maps[self.map_indices[model_name]][phi_id][src_theta_id][transform_id]
	--map = image.scale(map,self.loadSize[2], self.loadSize[2])
	--map = map:gt(0.3)
	if self.background == 1 then
		-- rgba -> rgb with random background
		im1_rgb = im1[{{1,3},{},{}}]
		im2_rgb = im2[{{1,3},{},{}}]
		local alpha1 = im1[4]:repeatTensor(3,1,1)
		local alpha2 = im2[4]:repeatTensor(3,1,1)
		
		local bgpath = self.data_dir .. '/background/' .. 
				string.format('%06d.jpg',torch.random(10000))
		local bg = image.load(bgpath, 3, 'float')
		if self.loadSize[2] < 256 then
			bg= image.scale(bg, self.loadSize[2], self.loadSize[2])
		end
		local bg_temp = (1-alpha1):cmul(bg)
		im1_rgb:cmul(alpha1):add(bg_temp)
		local bg_temp = (1-alpha2):cmul(bg)
		im2_rgb:cmul(alpha2):add(bg_temp)
	else
		-- rgba -> rgb with white background
		im1_rgb = im1[{{1,3},{},{}}]
		im2_rgb = im2[{{1,3},{},{}}]
		local alpha1 = im1[4]:repeatTensor(3,1,1)
		local alpha2 = im2[4]:repeatTensor(3,1,1)

		im1_rgb:cmul(alpha1):add(1-alpha1)
		im2_rgb:cmul(alpha2):add(1-alpha2)
	end
	return im1_rgb,im2_rgb,one_hot
end

return dataset
