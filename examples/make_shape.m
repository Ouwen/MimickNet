% Copyright Ouwen Huang 2019 

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at

%     http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function [image, image_height, image_width] = make_shape(image, divisible, custom_height, custom_width)
% make_shape(image, divisible, custom_height, custom_width) 
% makes an image divisible by some number
% image = the image to pad or crop
% divisible = the number to be divisible by (default 16)
% custom_height = height to nearest divisible by (defaults to image height)
% custom_width = width to nearest divisible by (defaults to image width)
%

image_height = size(image, 1);
image_width = size(image, 2);
if ~exist('divisible','var')
      divisible = 16;
end

if ~exist('custom_height','var')
      custom_height = image_height;
end

if ~exist('custom_width','var')
      custom_width = image_width;
end


if mod(custom_height, divisible) == 0
    height = custom_height;
else
    height = divisible - mod(custom_height, divisible) + custom_height;
end

if mod(custom_width, divisible) == 0
    width = custom_width;
else
    width = divisible - mod(custom_width, divisible) + custom_width;
end 

if image_height < height
    remainder = height - image_height;
    if mod(remainder, 2) == 0
        image = padarray(image,[remainder/2,0],'symmetric', 'both');
    else
        remainder = remainder - 1;
        image = padarray(image,[remainder/2,0],'symmetric', 'both');
        image = padarray(image,[1,0],'symmetric', 'pre');
    end
elseif image_height > height
    image = image(1:height, :);
end

if image_width < width
    remainder = width - image_width;
    if mod(remainder, 2) == 0
        image = padarray(image,[0, remainder/2],'symmetric', 'both');
    else
        remainder = remainder - 1;
        image = padarray(image,[0, remainder/2],'symmetric', 'both');
        image = padarray(image,[0,1],'symmetric', 'pre');
    end
elseif image_width > width
    image = image(:, 1:width);
end
