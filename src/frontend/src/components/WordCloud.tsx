import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import cloud from 'd3-cloud';
import inputData from '../data/most_similar_global_input.json';
import outputData from '../data/most_similar_global_output.json';

interface Word {
  text: string;
  size: number;
  x?: number;
  y?: number;
  rotate?: number;
}

interface D3WordCloudProps {
  selectedTokenIndex: number;
  selectedLayer: number;
  model: 'small' | 'large';
  variant: 'input' | 'output';
}

interface SimilarityData {
  [model: string]: {
    [layer: string]: {
      [tokenIndex: string]: {
        tokens: string[];
        similarities: number[];
      };
    };
  };
}

const D3WordCloud: React.FC<D3WordCloudProps> = ({ selectedTokenIndex, selectedLayer, model, variant }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const [words, setWords] = useState<Word[]>([]);

  useEffect(() => {
    const similarityData: SimilarityData = variant === 'input' ? inputData as SimilarityData : outputData as SimilarityData;
    const layerData = similarityData[model]?.[selectedLayer.toString()];
    const tokenData = layerData?.[selectedTokenIndex.toString()];

    if (tokenData) {
      const { tokens, similarities } = tokenData;
      const maxSimilarity = Math.max(...similarities);
      const minSimilarity = Math.min(...similarities);
      const newWords = tokens.map((token: string, index: number) => ({
        text: token,
        size: 10 + (similarities[index] - minSimilarity) / (maxSimilarity - minSimilarity) * 90,
      }));

      setWords(newWords);
    }
  }, [selectedTokenIndex, selectedLayer, model, variant]);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth;
        const height = width * 0.75; // 4:3 aspect ratio
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    if (!svgRef.current || words.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const layout = cloud<Word>()
      .size([dimensions.width, dimensions.height])
      .words(words)
      .padding(1)
      .rotate(() => 0)
      .font("Arial")
      .fontSize(d => d.size)
      .on("end", draw);

    layout.start();

    function draw(words: Word[]) {
      svg
        .append("g")
        .attr("transform", `translate(${dimensions.width / 2},${dimensions.height / 2})`)
        .selectAll("text")
        .data(words)
        .enter()
        .append("text")
        .style("font-size", d => `${d.size}px`)
        .style("font-family", "Arial")
        .attr("text-anchor", "middle")
        .attr("transform", d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
        .text(d => d.text)
        .on("mouseover", function() { d3.select(this).style("opacity", 0.7); })
        .on("mouseout", function() { d3.select(this).style("opacity", 1); });
    }
  }, [words, dimensions]);

  return (
    <div ref={containerRef} className="word-cloud-container w-full" style={{ paddingBottom: '75%', position: 'relative' }}>
      <svg 
        ref={svgRef} 
        width="100%" 
        height="100%" 
        viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
        preserveAspectRatio="xMidYMid meet"
        className="word-cloud-svg"
        aria-label="Word cloud visualization"
        style={{ position: 'absolute', top: 0, left: 0 }}
      ></svg>
    </div>
  );
};

export default D3WordCloud;