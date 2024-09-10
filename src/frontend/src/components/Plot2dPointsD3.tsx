import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group"
import { MoveIcon, HomeIcon, BoxSelect } from "lucide-react";

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

interface AdditionalPoints {
  tokens: string[];
  x: number[];
  y: number[];
}

interface Plot2dPointsD3Props {
  layer_idx: number;
}

const Plot2dPointsD3: React.FC<Plot2dPointsD3Props> = ({ layer_idx }) => {
  const containerRef = useRef<HTMLDivElement>(null);
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
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [additionalPoints, setAdditionalPoints] = useState<AdditionalPoints | null>(null);

  const aspectRatio = 3 / 2;
  const margin = { top: 20, right: 20, bottom: 30, left: 40 };

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth;
        const height = width / aspectRatio;
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    const fetchCloudData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/get_2d_cloud', {
          sample_rate: 0.1
        });
        setCloudData(response.data);
      } catch (error) {
        console.error('Error fetching cloud data:', error);
      }
    };

    fetchCloudData();
  }, []);

  const fetchAdditionalPoints = async () => {
    try {
      const response = await axios.post('http://localhost:8000/get_additional_points', {
        prompt: { text: "" },
        layer_idx: layer_idx
      });
      setAdditionalPoints(response.data);
    } catch (error) {
      console.error('Error fetching additional points:', error);
    }
  };

  useEffect(() => {
    fetchAdditionalPoints();
  }, [layer_idx]); // Fetch additional points when layer_idx changes

  useEffect(() => {
    if (!cloudData || !additionalPoints || !svgRef.current || dimensions.width === 0 || dimensions.height === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const x = d3.scaleLinear()
      .domain(d3.extent([...cloudData.x, ...additionalPoints.x]) as [number, number])
      .range([margin.left, dimensions.width - margin.right]);

    const y = d3.scaleLinear()
      .domain(d3.extent([...cloudData.y, ...additionalPoints.y]) as [number, number])
      .range([dimensions.height - margin.bottom, margin.top]);

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

    const addPointsWithProperties = (data: CloudPoint | AdditionalPoints, color: string) => {
      return g.selectAll(color === "red" ? ".additional-point" : "circle")
        .data(data.tokens)
        .join("circle")
        .attr("class", color === "red" ? "additional-point" : null)
        .attr("cx", (_, i) => x(data.x[i]))
        .attr("cy", (_, i) => y(data.y[i]))
        .attr("r", 3)
        .attr("fill", color)
        .attr("opacity", color === "red" ? 0.6 : 0.4)
        .on('mouseover', (event: MouseEvent, d: string) => {
          tooltip.style('visibility', 'visible')
            .text(d)
            .style('top', (event.pageY - 10) + 'px')
            .style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('visibility', 'hidden');
        });
    };

    const points = addPointsWithProperties(cloudData, "black");
    const additionalPointsSelection = addPointsWithProperties(additionalPoints, "red");

    // Add axes
    const xAxis = g.append('g')
      .attr('transform', `translate(0,${dimensions.height - margin.bottom})`)
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
        additionalPointsSelection.attr("r", 3 / transform.k); // Adjust additional points size
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    const brush = d3.brush()
      .extent([[0, 0], [dimensions.width, dimensions.height]])
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
      additionalPointsSelection.attr("r", 3 / currentZoomRef.current.k);
    };

    updateToolBehavior(activeTool);

    return () => {
      svg.on('.zoom', null);
      svg.on('.brush', null);
      if (brushRef.current) {
        brushG.call(brushRef.current.move, null);  // Clear brush on cleanup
      }
    };

  }, [cloudData, additionalPoints, activeTool, dimensions]);

  useEffect(() => {
    if (!brushExtent || !svgRef.current || !cloudData || !zoomRef.current) return;

    const svg = d3.select(svgRef.current);
    const [[x0, y0], [x1, y1]] = brushExtent;

    const transform = zoomTransform || d3.zoomIdentity;
    const xScale = transform.rescaleX(d3.scaleLinear()
      .domain(d3.extent(cloudData.x) as [number, number])
      .range([margin.left, dimensions.width - margin.right]));
    const yScale = transform.rescaleY(d3.scaleLinear()
      .domain(d3.extent(cloudData.y) as [number, number])
      .range([dimensions.height - margin.bottom, margin.top]));


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
      const padding = 20; // Reduced padding
      const xExtent = d3.extent(selectedX) as [number, number];
      const yExtent = d3.extent(selectedY) as [number, number];

      const xScale = d3.scaleLinear()
        .domain(d3.extent(cloudData.x) as [number, number])
        .range([margin.left, dimensions.width - margin.right]);

      const yScale = d3.scaleLinear()
        .domain(d3.extent(cloudData.y) as [number, number])
        .range([dimensions.height - margin.bottom, margin.top]);

      const x = (xExtent[0] + xExtent[1]) / 2;
      const y = (yExtent[0] + yExtent[1]) / 2;

      // Calculate scale based on the selection size relative to the SVG size
      const selectionWidth = Math.abs(x1 - x0);
      const selectionHeight = Math.abs(y1 - y0);
      const scaleX = (dimensions.width - padding * 2) / selectionWidth;
      const scaleY = (dimensions.height - padding * 2) / selectionHeight;
      const scale = Math.min(scaleX, scaleY);

      // Ensure the scale is not smaller than 1 (no zooming out)
      const limitedScale = Math.max(1, scale);

      const translate = [
        dimensions.width / 2 - limitedScale * xScale(x),
        dimensions.height / 2 - limitedScale * yScale(y)
      ];

      svg.transition().duration(750).call(
        zoomRef.current.transform,
        d3.zoomIdentity.translate(translate[0], translate[1]).scale(limitedScale)
      ).on("end", () => {
        // Clear the brush after zooming
        if (brushRef.current) {
          svg.select<SVGGElement>(".brush").call(brushRef.current.move as any, null);
        }
        currentZoomRef.current = d3.zoomIdentity.translate(translate[0], translate[1]).scale(limitedScale);
      });

      // Clear brush extent
      setBrushExtent(null);
    }

  }, [brushExtent, cloudData, zoomRef, dimensions]);

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

  if (!cloudData || dimensions.width === 0 || dimensions.height === 0) {
    return <div ref={containerRef} className="plot-container w-full" style={{ paddingBottom: `${100 / aspectRatio}%` }}>Loading...</div>;
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
          <BoxSelect className="h-4 w-4" />
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
          <HomeIcon className="h-4 w-4" />
        </ToggleGroupItem>
      </ToggleGroup>
      <div ref={containerRef} className="plot-container w-full" style={{ paddingBottom: `${100 / aspectRatio}%`, position: 'relative' }}>
        <svg 
          ref={svgRef}
          width="100%" 
          height="100%" 
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ position: 'absolute', top: 0, left: 0 }}
        ></svg>
      </div>
    </div>
  );
};

export default Plot2dPointsD3;