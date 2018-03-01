/*
 * maxflow.h
 *
 *  Created on: 10 de fev de 2017
 *      Author: ivisionlab
 */

#ifndef LIB_GCO_INCLUDE_MAXFLOW_H_
#define LIB_GCO_INCLUDE_MAXFLOW_H_




#include <stdio.h>
#include "graph.h"


/*
	special constants for node->parent
*/
#define TERMINAL ( (arc *) 1 )		/* to terminal */
#define ORPHAN   ( (arc *) 2 )		/* orphan */


#define INFINITE_D ((int)(((unsigned)-1)/2))		/* infinite distance to the terminal */

/*
template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_active(node *i);

template <typename captype, typename tcaptype, typename flowtype>
	inline typename Graph<captype,tcaptype,flowtype>::node* Graph<captype,tcaptype,flowtype>::next_active();

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_orphan_front(node *i);

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::set_orphan_rear(node *i);

template <typename captype, typename tcaptype, typename flowtype>
	inline void Graph<captype,tcaptype,flowtype>::add_to_changed_list(node *i);


template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::maxflow_init();

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::maxflow_reuse_trees_init();

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::augment(arc *middle_arc);

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::process_source_orphan(node *i);

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::process_sink_orphan(node *i);

template <typename captype, typename tcaptype, typename flowtype>
	flowtype Graph<captype,tcaptype,flowtype>::maxflow(bool reuse_trees, Block<node_id>* _changed_list);

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::test_consistency(node* current_node);

template <typename captype, typename tcaptype, typename flowtype>
	void Graph<captype,tcaptype,flowtype>::Copy(Graph<captype, tcaptype, flowtype>* g0);
*/


#endif /* LIB_GCO_INCLUDE_MAXFLOW_H_ */
