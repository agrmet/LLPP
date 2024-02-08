//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>

using namespace std;

namespace Ped {
	class Twaypoint;
	// ---------------------- Tagent ---------------------
	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; }
		int getDesiredY() const { return desiredPositionY; }

		// Sets the agent's position
		void setX(int newX) { x = newX; }
		void setY(int newY) { y = newY; }

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		// Position of agent defined by x and y
		int getX() const { return x; };
		int getY() const { return y; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		// Retrieves the agent's waypoints
		deque<Twaypoint*> getWaypoints() { return waypoints; };

	private:
		Tagent() {};

		// The agent's current position
		int x;
		int y;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The current destination (may require several steps to reach)
		Twaypoint* destination;

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;

		// Internal init function 
		void init(int posX, int posY);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();
	};

	// ---------------------- TagentSoA ---------------------
	class TagentSoA {
		public: 
		TagentSoA() {}
		// Sets the attributes for all agents (xP, yP, xDesP, yDesP, waypoints)
		TagentSoA(std::vector<Tagent*> &agentsInScenario);
	
		// Arrays of all agents' relevant attributes
		// xP: current x coordinates 
		// yP: current y coordinates
		// xDesP: desired x coordinates
		// yDesP: desired y coordinates
		std::vector<float> xP;
		std::vector<float> yP;
		std::vector<float> xDesP;
		std::vector<float> yDesP;

		// Waypoint attributes:
		// xWP: all agents' x coordinate-waypoints
		// yWP: all agents' y coordinate-waypoints
		// id: all agents' waypoint-id's
		// r: all agents' waypoint-radius
		std::vector<float> xWP;
		std::vector<float> yWP;
		std::vector<float> id;
		std::vector<float> r;

		// The queue of destinations that all agents still has to visit
		std::vector< deque<Twaypoint*> > waypointsAll;

		// Vectorized functions:
		void computeNextPositionsVectorized(int idx);

	};
}


#endif