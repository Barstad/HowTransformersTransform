import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import cloud from 'd3-cloud';
import axios from 'axios';

interface Word {
  text: string;
  size: number;
  x?: number;
  y?: number;
  rotate?: number;
}

interface D3WordCloudProps {
  selectedTokenIndex: number;
  width: number;
  height: number;
}

const D3WordCloud: React.FC<D3WordCloudProps> = ({ selectedTokenIndex, width, height }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [words, setWords] = useState<Word[]>([]);

  useEffect(() => {
    const fetchWords = async () => {
      try {
        const response = await axios.post('http://localhost:8000/get_most_similar_global', {
          token_idx: selectedTokenIndex,
          prompt: { text: '' },
          num_tokens: 50
        });

        const { tokens, similarities } = response.data;
        const maxSimilarity = Math.max(...similarities);
        const minSimilarity = Math.min(...similarities);
        const newWords = tokens.map((token: string, index: number) => ({
          text: token,
          size: 10 + (similarities[index] - minSimilarity) / (maxSimilarity - minSimilarity) * 90,
        }));

        setWords(newWords);
      } catch (error) {
        console.error('Error fetching word cloud data:', error);
      }
    };

    fetchWords();
  }, [selectedTokenIndex]);

  useEffect(() => {
    if (!svgRef.current || words.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const layout = cloud<Word>()
      .size([width, height])
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
        .attr("transform", `translate(${width / 2},${height / 2})`)
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
  }, [words, width, height]);

  return (
    <div className="word-cloud-container h-full">
      <svg 
        ref={svgRef} 
        width="100%" 
        height="100%" 
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="word-cloud-svg"
        aria-label="Word cloud visualization"
      ></svg>
    </div>
  );
};

export default D3WordCloud;