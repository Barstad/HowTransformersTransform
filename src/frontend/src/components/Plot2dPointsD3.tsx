import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group"
import { MoveIcon, MousePointerClick, ZoomOutIcon } from "lucide-react";
import { brushSelection } from 'd3';

interface CloudPoint {
  tokens: string[];
  x: number[];
  y: number[];
}

interface SelectedPoints {
  tokens: string[];
  indices: number[];
  x: number[];
  y: number[];
}

const Plot2dPointsD3: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const brushRef = useRef<d3.BrushBehavior<unknown> | null>(null);
  const [cloudData, setCloudData] = useState<CloudPoint | null>(null);
  const [zoomTransform, setZoomTransform] = useState<d3.ZoomTransform | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<SelectedPoints>({ tokens: [], indices: [] , x: [], y: []});
  const [isSelectionEnabled, setIsSelectionEnabled] = useState(false);
  const [activeTool, setActiveTool] = useState<string>("move");
  const [brushExtent, setBrushExtent] = useState<[[number, number], [number, number]] | null>(null);
  const currentZoomRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);

  const width = 600;
  const height = 400;
  const margin = { top: 20, right: 20, bottom: 30, left: 40 };

  useEffect(() => {
    const fetchCloudData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/get_2d_cloud', {
          page: 1,
          page_size: 500
        });
        setCloudData(response.data);
      } catch (error) {
        console.error('Error fetching cloud data:', error);
      }
    };

    fetchCloudData();
  }, []);

  useEffect(() => {
    if (!cloudData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const x = d3.scaleLinear()
      .domain(d3.extent(cloudData.x) as [number, number])
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain(d3.extent(cloudData.y) as [number, number])
      .range([height - margin.bottom, margin.top]);

    const g = svg.append('g');

    // Create tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'white')
      .style('border', 'solid')
      .style('border-width', '1px')
      .style('border-radius', '5px')
      .style('padding', '10px');

    const points = g.selectAll("circle")
      .data(cloudData.tokens)
      .join("circle")
      .attr("cx", (_, i) => x(cloudData.x[i]))
      .attr("cy", (_, i) => y(cloudData.y[i]))
      .attr("r", 3)
      .attr("fill", "black")
      .attr("opacity", 0.4)
      .on('mouseover', (event, d) => {
        tooltip.style('visibility', 'visible')
          .text(d)
          .style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    // Add axes
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x));

    const yAxis = g.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 20])
      .on('zoom', (event) => {
        const transform = event.transform;
        setZoomTransform(transform);
        currentZoomRef.current = transform; // Update currentZoom ref
        g.attr('transform', transform.toString());
        xAxis.call(d3.axisBottom(transform.rescaleX(x)));
        yAxis.call(d3.axisLeft(transform.rescaleY(y)));
        points.attr("r", 3 / transform.k); // Adjust the radius of the points based on the zoom levelS
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    const brush = d3.brush()
      .extent([[0, 0], [width, height]])
      .on("end", (event) => {
        if (!event.selection) {
          setBrushExtent(null);
          return;
        }
        
        setBrushExtent(event.selection as [[number, number], [number, number]]);
        console.log(brushExtent)
      });

    brushRef.current = brush;

    const brushG = svg.append("g")
      .attr("class", "brush")
      .style("display", "none");

    brushG.call(brush);

    const updateToolBehavior = (tool: string) => {
      if (tool === "select") {
        brushG.style('display', null);
        svg.on('.zoom', null);
        // Reset and re-initialize the brush
        brushG.call(brush.move, null);
        brushG.call(brush);
      } else {
        brushG.style('display', 'none');
        svg.call(zoom);
      }
      // Apply current zoom transform regardless of the selected tool
      g.attr('transform', currentZoomRef.current.toString());
      xAxis.call(d3.axisBottom(currentZoomRef.current.rescaleX(x)));
      yAxis.call(d3.axisLeft(currentZoomRef.current.rescaleY(y)));
      points.attr("r", 3 / currentZoomRef.current.k);
    };

    updateToolBehavior(activeTool);

    return () => {
      svg.on('.zoom', null);
      svg.on('.brush', null);
      if (brushRef.current) {
        brushG.call(brushRef.current.move, null);  // Clear brush on cleanup
      }
    };

  }, [cloudData, activeTool]);

  useEffect(() => {
    if (!brushExtent || !svgRef.current || !cloudData || !zoomRef.current) return;

    const svg = d3.select(svgRef.current);
    const [[x0, y0], [x1, y1]] = brushExtent;

    const transform = zoomTransform || d3.zoomIdentity;
    const xScale = transform.rescaleX(d3.scaleLinear()
      .domain(d3.extent(cloudData.x) as [number, number])
      .range([margin.left, width - margin.right]));
    const yScale = transform.rescaleY(d3.scaleLinear()
      .domain(d3.extent(cloudData.y) as [number, number])
      .range([height - margin.bottom, margin.top]));


    const selectedIndices: number[] = [];
    const selectedTokens: string[] = [];
    const selectedX: number[] = [];
    const selectedY: number[] = [];

    svg.selectAll("circle")
      .attr("fill", (_, i) => {
        const cx = xScale(cloudData.x[i]);
        const cy = yScale(cloudData.y[i]);
        if (x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1) {
          selectedIndices.push(i);
          selectedTokens.push(cloudData.tokens[i]);
          selectedX.push(cloudData.x[i]);
          selectedY.push(cloudData.y[i]);
          return "orange";
        }
        return "black";
      });

    setSelectedPoints({ tokens: selectedTokens, indices: selectedIndices, x: selectedX, y: selectedY });

    // Zoom to the selected region
    if (selectedX.length > 0 && selectedY.length > 0) {
      const padding = 50; // Padding around the selected area
      const xExtent = d3.extent(selectedX) as [number, number];
      const yExtent = d3.extent(selectedY) as [number, number];

      const xScale = d3.scaleLinear()
        .domain(d3.extent(cloudData.x) as [number, number])
        .range([margin.left, width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain(d3.extent(cloudData.y) as [number, number])
        .range([height - margin.bottom, margin.top]);

      const xRange = xScale.range();
      const yRange = yScale.range();

      const dx = xExtent[1] - xExtent[0];
      const dy = yExtent[1] - yExtent[0];
      const x = (xExtent[0] + xExtent[1]) / 2;
      const y = (yExtent[0] + yExtent[1]) / 2;
      const scale = Math.min(
        0.9 / Math.max(dx / (xRange[1] - xRange[0]), dy / (yRange[0] - yRange[1])),
        8
      );
      const translate = [width / 2 - scale * xScale(x), height / 2 - scale * yScale(y)];

      svg.transition().duration(750).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
      ).on("end", () => {
        // Clear the brush after zooming
        if (brushRef.current) {
          svg.select<SVGGElement>(".brush").call(brushRef.current.move as any, null);
        }
        currentZoomRef.current = d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale); // Update currentZoom ref
      });

      // Clear brush extent
      setBrushExtent(null);
    }

  }, [brushExtent, cloudData, zoomRef]);

  console.log(selectedPoints)

  const handleReset = () => {
    if (svgRef.current && zoomRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(750).call(zoomRef.current.transform, d3.zoomIdentity);
      setZoomTransform(null);
      currentZoomRef.current = d3.zoomIdentity; // Reset currentZoom ref
    }
  };

  const handleToolChange = (value: string) => {
    if (value) {
      setActiveTool(value);
      if (value === "select") {
        setIsSelectionEnabled(true);
        // Clear any existing brush when switching to select tool
        if (svgRef.current && brushRef.current) {
          const svg = d3.select(svgRef.current);
          svg.select<SVGGElement>(".brush").call(brushRef.current.move as any, null);
        }
      } else {
        setIsSelectionEnabled(false);
      }
    }
  };

  if (!cloudData) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <ToggleGroup type="single" value={activeTool} onValueChange={handleToolChange}>
        <ToggleGroupItem 
          value="move" 
          aria-label="Move" 
          style={{
            padding: '8px',
            backgroundColor: activeTool === "move" ? '#4A5568' : 'transparent',
            color: activeTool === "move" ? 'white' : 'black'
          }}
        >
          <MoveIcon className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem 
          value="select" 
          aria-label="Select" 
          style={{
            padding: '8px',
            backgroundColor: activeTool === "select" ? '#4A5568' : 'transparent',
            color: activeTool === "select" ? 'white' : 'black'
          }}
        >
          <MousePointerClick className="h-4 w-4" />
        </ToggleGroupItem>
        <ToggleGroupItem 
          value="reset" 
          aria-label="Reset Zoom" 
          onClick={handleReset}
          style={{
            padding: '8px',
            backgroundColor: activeTool === "reset" ? '#4A5568' : 'transparent',
            color: activeTool === "reset" ? 'white' : 'black'
          }}
        >
          <ZoomOutIcon className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>
      <div style={{ border: '1px solid black', width: `${width}px`, height: `${height}px` }}>
        <svg ref={svgRef} width={width} height={height}></svg>
      </div>
      <div>
        Selected points: {selectedPoints.tokens.join(', ')}
      </div>
    </div>
  );
};

export default Plot2dPointsD3;