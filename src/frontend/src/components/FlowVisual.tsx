import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import cloud from 'd3-cloud';
import inputData from '../data/most_similar_global_input.json';
import outputData from '../data/most_similar_global_output.json';

interface FlowVisualProps {
    selectedTokenIndex: number;
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

interface TokenNode extends d3.SimulationNodeDatum {
  token: string;
  similarity: number;
}

const FlowVisual: React.FC<FlowVisualProps> = ({ selectedTokenIndex, model, variant }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const nodeRadius = 120;
  const nodesPerRow = 4;
  const margin = { top: nodeRadius + 30, right: 120, bottom: nodeRadius + 30, left: 120 };
  const minVerticalSpacing = nodeRadius * 3; // Increased minimum vertical spacing

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        console.log('Container dimensions:', width, height);
        setDimensions({ width, height });
      }
    };

    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    updateDimensions();

    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    if (!svgRef.current || dimensions.width === 0 || dimensions.height === 0) {
      console.log('SVG ref or dimensions not ready:', svgRef.current, dimensions);
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const similarityData: SimilarityData = (variant === 'input' ? inputData : outputData) as SimilarityData;
    const modelData = similarityData[model] || {};
    const layers = Object.keys(modelData).sort((a, b) => parseInt(a) - parseInt(b));

    console.log('Layers:', layers);

    // Calculate the number of rows based on the number of layers and nodes per row
    const totalRows = Math.ceil(layers.length / nodesPerRow);

    // Calculate vertical spacing
    const availableHeight = dimensions.height - margin.top - margin.bottom;
    const verticalSpacing = Math.max(minVerticalSpacing, availableHeight / totalRows);

    // Recalculate total height needed
    const totalHeight = verticalSpacing * totalRows + margin.top + margin.bottom;

    console.log('SVG dimensions:', dimensions.width, totalHeight);

    // Update SVG viewBox to accommodate all nodes
    svg.attr("width", dimensions.width)
       .attr("height", totalHeight)
       .attr("viewBox", `0 0 ${dimensions.width} ${totalHeight}`);

    // Debug rectangle
    svg.append("rect")
       .attr("width", dimensions.width)
       .attr("height", totalHeight)
       .attr("fill", "none")
       .attr("stroke", "red");

    const getNodeCoordinates = (index: number) => {
      const row = Math.floor(index / nodesPerRow);
      const col = index % nodesPerRow;
      const x = margin.left + (row % 2 === 0 ? col : (nodesPerRow - 1 - col)) * ((dimensions.width - margin.left - margin.right) / (nodesPerRow - 1));
      const y = margin.top + (row + 0.5) * verticalSpacing;
      return { x, y };
    };

    // Create path
    const lineGenerator = d3.line<{ x: number; y: number }>()
      .x(d => d.x)
      .y(d => d.y)
      .curve(d3.curveLinear);

    const pathData = layers.map((_, i) => getNodeCoordinates(i));

    svg.append("path")
      .datum(pathData)
      .attr("fill", "none")
      .attr("stroke", "#ccc")
      .attr("stroke-width", 2)
      .attr("d", lineGenerator);

    // Create nodes and word clouds
    layers.forEach((layer, i) => {
      const { x, y } = getNodeCoordinates(i);
      const layerData = modelData[layer]?.[selectedTokenIndex.toString()];

      if (!layerData || !layerData.tokens || !layerData.similarities) {
        console.error(`Missing data for layer ${layer}, token index ${selectedTokenIndex}`);
        console.log('modelData:', modelData);
        console.log('layerData:', layerData);
        return; // Skip this iteration
      }

      // Create node
      const nodeGroup = svg.append("g")
        .attr("transform", `translate(${x},${y})`);

      // Add a blurred gray circle
      nodeGroup.append("circle")
        .attr("r", nodeRadius)
        .attr("fill", "rgba(200, 200, 200, 0.3)")
        .attr("filter", "url(#blur)");

      // Add a smaller, sharper circle on top for better contrast
      nodeGroup.append("circle")
        .attr("r", nodeRadius * 0.9)
        .attr("fill", "rgba(220, 220, 220, 0.7)");

      // Create word cloud
      const words = layerData.tokens.map((token: string, index: number) => ({
        text: token,
        size: 10 + (layerData.similarities[index] - Math.min(...layerData.similarities)) / 
              (Math.max(...layerData.similarities) - Math.min(...layerData.similarities)) * 20
      }));

      const layout = cloud()
        .size([nodeRadius * 2, nodeRadius * 2])
        .words(words)
        .padding(3)
        .rotate(() => 0)
        .font("Arial")
        .fontSize(d => d.size)
        .on("end", draw);

      layout.start();

      function draw(words: { text: string; size: number; x?: number; y?: number; rotate?: number }[]) {
        const cloudGroup = nodeGroup.append("g");

        cloudGroup.selectAll("text")
          .data(words)
          .enter().append("text")
          .style("font-size", d => `${d.size}px`)
          .style("font-family", "Arial")
          .attr("text-anchor", "middle")
          .attr("transform", d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
          .text(d => d.text)
          .attr("opacity", 0);

        // Get the bounding box of the entire word cloud
        const bbox = cloudGroup.node()!.getBBox();

        // Calculate the scale to fit the word cloud in the node
        const scale = Math.min(
          (nodeRadius * 2) / bbox.width,
          (nodeRadius * 2) / bbox.height
        ) * 0.9; // 0.9 to add a small margin

        // Apply the transformation to center and scale the word cloud
        cloudGroup.attr("transform", `translate(${nodeRadius},${nodeRadius}) scale(${scale}) translate(${-bbox.width/2},${-bbox.height/2})`);

        cloudGroup.selectAll("text")
          .on("mouseover", function() { d3.select(this).style("opacity", 0.7); })
          .on("mouseout", function() { d3.select(this).style("opacity", 1); })
          .transition()
          .delay((_, i) => i * 50)
          .duration(500)
          .attr("opacity", 1);
      }

      // Layer label
      svg.append("text")
        .attr("x", x)
        .attr("y", y + nodeRadius + 20)
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .attr("fill", "#333")
        .text(`Layer ${layer}`);
    });

    // Add blur filter definition
    svg.append("defs")
      .append("filter")
      .attr("id", "blur")
      .append("feGaussianBlur")
      .attr("stdDeviation", 3);

    // Title
    svg.append("text")
      .attr("x", dimensions.width / 2)
      .attr("y", margin.top / 2)  // Position the title in the middle of the top margin
      .attr("text-anchor", "middle")
      .attr("font-size", "20px")
      .attr("font-weight", "bold")
      .text(`Token Evolution Timeline - ${model.toUpperCase()} Model (${variant.charAt(0).toUpperCase() + variant.slice(1)})`);

    console.log('Rendering complete');

  }, [selectedTokenIndex, model, variant, dimensions, margin.top, margin.right, margin.bottom, margin.left]);

  return (
    <div ref={containerRef} className="flow-visual-container w-full h-full" style={{ position: 'relative', minHeight: '500px' }}>
      <svg 
        ref={svgRef} 
        width="100%" 
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        className="flow-visual-svg"
        aria-label="Flow visualization"
        style={{ position: 'absolute', top: 0, left: 0 }}
      ></svg>
    </div>
  );
};

export default FlowVisual;