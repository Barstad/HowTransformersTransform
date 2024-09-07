import React from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';

interface LayerMenuProps {
  onLayerSelect: (layer: number) => void;
}

const LayerMenu: React.FC<LayerMenuProps> = ({ onLayerSelect }) => {
  const layers = Array.from({ length: 33 }, (_, i) => i);

  return (
    <div className="layer-menu">
      <Select onValueChange={(value) => onLayerSelect(Number(value))}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select a layer" />
        </SelectTrigger>
        <SelectContent className="bg-white dark:bg-gray-800">
          {layers.map((layer) => (
            <SelectItem key={layer} value={layer.toString()}>
              {layer === 0 ? "Embedding Table" : `Layer ${layer}`}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default LayerMenu;
