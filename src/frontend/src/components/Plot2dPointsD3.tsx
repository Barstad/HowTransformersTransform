import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

interface CloudPoint {
  tokens: string[];
  x: number[];
  y: number[];
}

const Plot2dPointsD3: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [cloudData, setCloudData] = useState<CloudPoint | null>(null);
  const [zoomTransform, setZoomTransform] = useState<d3.ZoomTransform | null>(null);

  const width = 800;
  const height = 600;
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
      .attr("fill", "blue")
      .attr("opacity", 0.6)
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

    // Zoom function
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 20])
      .on('zoom', (event) => {
        const transform = event.transform;
        setZoomTransform(transform);
        g.attr('transform', transform.toString());
        xAxis.call(d3.axisBottom(transform.rescaleX(x)));
        yAxis.call(d3.axisLeft(transform.rescaleY(y)));
        
        // Adjust point size based on zoom level
        const newRadius = 3 / transform.k;
        points.attr("r", newRadius);
      });

    svg.call(zoom);

    // Store the zoom object in the ref
    zoomRef.current = zoom;

  }, [cloudData]);

  const handleReset = () => {
    if (svgRef.current && zoomRef.current) {
      const svg = d3.select(svgRef.current);
      svg.transition().duration(750).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  };

  if (!cloudData) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <div style={{ border: '1px solid black', width: `${width}px`, height: `${height}px` }}>
        <svg ref={svgRef} width={width} height={height}></svg>
      </div>
      <button onClick={handleReset} style={{ marginTop: '10px' }}>Reset Zoom</button>
    </div>
  );
};

export default Plot2dPointsD3;