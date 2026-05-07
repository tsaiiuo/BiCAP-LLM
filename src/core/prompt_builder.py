"""
Traffic Prompt Builder for Flow Forecasting

Generates semantic text prompts to enhance an LLM's understanding of traffic patterns.
Default level: 'simple' (time + traffic period)
"""

import torch


class TrafficPromptBuilder:
    """
    Build traffic-aware text prompts for the LLM backbone.

    Levels:
    - 'simple': Time + traffic period (e.g., "Mon 08:00 morning rush")
    - 'enhanced': Detailed traffic semantics with expectations
    - 'task': Task-oriented description (e.g., "Given historical traffic... predict...")
    """

    def __init__(self, level='simple', num_nodes=307, L=12, P=12):
        """
        Args:
            level: 'simple', 'enhanced', or 'task'
            num_nodes: Number of sensor nodes in the graph (for task prompts)
            L: Historical sequence length (for task prompts)
            P: Prediction horizon length (for task prompts)
        """
        self.level = level
        self.num_nodes = num_nodes
        self.L = L
        self.P = P
        print(f"Traffic Prompt Builder initialized (level: {level})")

    def generate_batch(self, timestamp, input_data=None):
        """
        Build text prompts for a batch of samples.

        Args:
            timestamp: (B, T, 5) - [year, month, day_of_week, hour, minute]
                      or (B, T, 4) - [year, month, day_of_week, hour]
            input_data: (B, T, N, F) - Optional input traffic data for data-driven prompts

        Returns:
            List[str] of length B
        """
        B = timestamp.size(0)
        T = timestamp.size(1)
        prompts = []

        for i in range(B):
            # Calculate statistics if input_data is provided
            stats = None
            if input_data is not None and self.level == 'task_enhanced':
                stats = self._calculate_stats(input_data[i])

            if self.level == 'task' or self.level == 'task_enhanced':
                # For task prompts, need start and end timestamps
                ts_start = timestamp[i, 0, :]  # First timestep
                ts_end = timestamp[i, -1, :]   # Last timestep

                # Extract time info
                day_of_week = int(ts_end[2].item()) % 7
                hour_start = int(ts_start[3].item()) % 24
                minute_start = int(ts_start[4].item()) % 60 if ts_start.size(0) > 4 else 0
                hour_end = int(ts_end[3].item()) % 24
                minute_end = int(ts_end[4].item()) % 60 if ts_end.size(0) > 4 else 0

                if self.level == 'task_enhanced':
                    prompt = self._generate_task_enhanced_prompt(
                        day_of_week, hour_start, minute_start, hour_end, minute_end, stats
                    )
                else:
                    prompt = self._generate_task_prompt(
                        day_of_week, hour_start, minute_start, hour_end, minute_end
                    )
            else:
                # For simple/enhanced, just use last timestep
                ts = timestamp[i, -1, :]
                day_of_week = int(ts[2].item()) % 7
                hour = int(ts[3].item()) % 24
                prompt = self._generate_single_prompt(day_of_week, hour)

            prompts.append(prompt)

        return prompts

    def _generate_single_prompt(self, day_of_week, hour):
        """
        Build a prompt for a single sample.

        Args:
            day_of_week: 0-6 (0=Monday, 6=Sunday)
            hour: 0-23

        Returns:
            str: Text prompt
        """
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_name = day_names[day_of_week]

        if self.level == 'simple':
            # Format: "Mon 08:00 morning rush"
            period = self._get_traffic_period(day_of_week, hour)
            return f"{day_name} {hour:02d}:00 {period}"

        elif self.level == 'enhanced':
            # Format: "Mon 08:00: morning rush hour, expect high traffic"
            description = self._get_detailed_description(day_of_week, hour)
            return f"{day_name} {hour:02d}:00: {description}"

        else:
            raise ValueError(f"Unknown prompt level: {self.level}")

    def _get_traffic_period(self, day_of_week, hour):
        """
        Determine traffic period label (for simple prompts).

        Returns short labels like "morning rush", "evening rush", etc.
        """
        # Weekend
        if day_of_week >= 5:  # Sat, Sun
            if 10 <= hour <= 20:
                return "weekend leisure"
            else:
                return "weekend quiet"

        # Weekday
        if 6 <= hour <= 9:
            return "morning rush"
        elif 17 <= hour <= 19:
            return "evening rush"
        elif 10 <= hour <= 16:
            return "midday normal"
        elif 20 <= hour <= 22:
            return "evening normal"
        else:  # 23-5
            return "late night"

    def _get_detailed_description(self, day_of_week, hour):
        """
        Produce detailed traffic description (for enhanced prompts).

        Returns longer descriptions with traffic expectations.
        """
        # Weekend
        if day_of_week >= 5:  # Sat, Sun
            if 10 <= hour <= 14:
                return "weekend leisure travel, moderate to high traffic flow"
            elif 15 <= hour <= 20:
                return "weekend afternoon, steady traffic for shopping and dining"
            else:
                return "weekend quiet hours, low traffic flow"

        # Weekday
        if 6 <= hour <= 9:
            return "morning rush hour, expect high traffic and congestion"
        elif 17 <= hour <= 19:
            return "evening commute peak, heavy traffic likely with delays"
        elif 10 <= hour <= 16:
            return "normal business hours, steady traffic flow"
        elif 20 <= hour <= 22:
            return "evening hours, moderate traffic decreasing"
        else:  # 23-5
            return "late night or early morning, minimal traffic flow"

    def _generate_task_prompt(self, day_of_week, hour_start, minute_start, hour_end, minute_end):
        """
        Build a task-oriented prompt (for 'task' level).

        Format: "Given the historical traffic values for N nodes from HH:MM to HH:MM on Day.
                 Your task is to predict the traffic values for the next M hours."

        Args:
            day_of_week: 0-6
            hour_start, minute_start: Start time
            hour_end, minute_end: End time

        Returns:
            str: Task-oriented prompt
        """
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week]

        # Format times
        start_time_str = f"{hour_start:02d}:{minute_start:02d}"
        end_time_str = f"{hour_end:02d}:{minute_end:02d}"

        # Calculate prediction duration (in hours)
        # Assuming P timesteps at 5-min intervals
        predict_minutes = self.P * 5
        if predict_minutes >= 60:
            predict_duration = f"{predict_minutes // 60} hour" + ("s" if predict_minutes >= 120 else "")
        else:
            predict_duration = f"{predict_minutes} minutes"

        prompt = (
            f"Given the historical traffic values for {self.num_nodes} nodes "
            f"from {start_time_str} to {end_time_str} on {day_name}. "
            f"Your task is to predict the traffic values for the next {predict_duration}."
            f"The historical traffic values of each node are as follows:"
        )

        return prompt

    def _calculate_stats(self, input_data):
        """
        Compute statistics from input traffic data.

        Args:
            input_data: (T, N, F) - Traffic data for one sample

        Returns:
            dict with statistics
        """
        import torch

        # Remove masked values (zeros) for better statistics
        valid_data = input_data[input_data > 0]

        if len(valid_data) == 0:
            # All zeros, return defaults
            return {
                'mean': 0.0,
                'trend': 0.0,
                'volatility': 'stable'
            }

        # Calculate mean flow across all nodes and timesteps
        mean_flow = valid_data.mean().item()

        # Calculate trend (compare first half vs second half)
        T = input_data.size(0)
        mid = T // 2
        first_half = input_data[:mid][input_data[:mid] > 0]
        second_half = input_data[mid:][input_data[mid:] > 0]

        if len(first_half) > 0 and len(second_half) > 0:
            trend = second_half.mean().item() - first_half.mean().item()
        else:
            trend = 0.0

        # Calculate volatility
        if len(valid_data) > 1:
            std = valid_data.std().item()
            cv = std / (mean_flow + 1e-6)  # Coefficient of variation

            if cv < 0.15:
                volatility = 'stable'
            elif cv < 0.30:
                volatility = 'moderate'
            else:
                volatility = 'high'
        else:
            volatility = 'stable'

        return {
            'mean': mean_flow,
            'trend': trend,
            'volatility': volatility
        }

    def _generate_task_enhanced_prompt(self, day_of_week, hour_start, minute_start,
                                       hour_end, minute_end, stats):
        """
        Build enhanced task-oriented prompt with data-driven observations.

        Structure:
        1. Context (time, day)
        2. Pattern Recognition (traffic period description)
        3. Data Observations (current flow, trend)
        4. Task + Reasoning Guidance

        Args:
            day_of_week: 0-6
            hour_start, minute_start: Start time
            hour_end, minute_end: End time
            stats: dict with traffic statistics

        Returns:
            str: Enhanced task-oriented prompt
        """
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week]

        # Format times
        start_time_str = f"{hour_start:02d}:{minute_start:02d}"
        end_time_str = f"{hour_end:02d}:{minute_end:02d}"

        # Calculate prediction duration
        predict_minutes = self.P * 5
        if predict_minutes >= 60:
            predict_duration = f"{predict_minutes // 60} hour" + ("s" if predict_minutes >= 120 else "")
        else:
            predict_duration = f"{predict_minutes} minutes"

        # Stage 1: Temporal Context
        context = f"Time: {day_name} {start_time_str} to {end_time_str}. "

        # Stage 2: Pattern Recognition (traffic period)
        period = self._get_traffic_period(day_of_week, hour_start)

        if period == "morning rush":
            pattern = ("This is morning rush hour. Traffic typically increases rapidly "
                      "as commuters travel to work, with peak congestion expected. ")
        elif period == "evening rush":
            pattern = ("This is evening commute period. High traffic volume expected "
                      "with congestion on major routes as people return home. ")
        elif period == "weekend leisure":
            pattern = ("This is weekend leisure time. Moderate traffic expected "
                      "with shopping and recreational travel patterns. ")
        elif period == "midday normal":
            pattern = ("This is midday period. Steady business-hour traffic "
                      "with moderate flow on main routes. ")
        elif period == "late night":
            pattern = ("This is late night or early morning. Minimal traffic flow "
                      "with mostly freight and essential travel. ")
        else:
            pattern = f"This is {period} period. "

        # Stage 3: Data-driven Observations
        if stats is not None:
            mean_flow = stats['mean']
            trend = stats['trend']
            volatility = stats['volatility']

            # Format observations
            if mean_flow > 0:
                obs_flow = f"Current average flow: {mean_flow:.1f} vehicles. "
            else:
                obs_flow = "Limited traffic data available. "

            if trend > 5:
                obs_trend = "Trend: increasing rapidly. "
            elif trend > 1:
                obs_trend = "Trend: gradually increasing. "
            elif trend < -5:
                obs_trend = "Trend: decreasing rapidly. "
            elif trend < -1:
                obs_trend = "Trend: gradually decreasing. "
            else:
                obs_trend = "Trend: stable. "

            obs_volatility = f"Volatility: {volatility}. "

            observation = obs_flow + obs_trend + obs_volatility
        else:
            observation = ""

        # Stage 4: Task + Reasoning Guidance
        task = (
            f"Your task is to predict traffic flow for {self.num_nodes} nodes "
            f"over the next {predict_duration}. "
            f"Consider: (1) {period} patterns, (2) current trend direction, "
            f"(3) typical {day_name} behavior, (4) flow volatility."
        )

        # Combine all stages
        enhanced_prompt = context + pattern + observation + task

        return enhanced_prompt

    def _get_traffic_period(self, day_of_week, hour):
        """
        Determine traffic period label (reused from simple prompts).
        """
        # Weekend
        if day_of_week >= 5:  # Sat, Sun
            if 10 <= hour <= 20:
                return "weekend leisure"
            else:
                return "weekend quiet"

        # Weekday
        if 6 <= hour <= 9:
            return "morning rush"
        elif 17 <= hour <= 19:
            return "evening rush"
        elif 10 <= hour <= 16:
            return "midday normal"
        elif 20 <= hour <= 22:
            return "evening normal"
        else:  # 23-5
            return "late night"


# Test
if __name__ == "__main__":
    print("Testing TrafficPromptBuilder...")

    # Create builder
    gen = TrafficPromptBuilder(level='simple')

    # Test cases
    test_cases = [
        # [year, month, day_of_week, hour, minute]
        [2024, 1, 0, 8, 30],   # Monday 8:30 AM
        [2024, 1, 0, 18, 0],   # Monday 6:00 PM
        [2024, 1, 4, 18, 0],   # Friday 6:00 PM
        [2024, 1, 5, 14, 0],   # Saturday 2:00 PM
        [2024, 1, 6, 2, 0],    # Sunday 2:00 AM
    ]

    # Create dummy timestamp tensor
    timestamp = torch.tensor(test_cases).unsqueeze(1)  # (5, 1, 5)

    # Generate prompts
    prompts = gen.generate_batch(timestamp)

    print("\nSimple prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")

    # Test enhanced
    gen_enhanced = TrafficPromptBuilder(level='enhanced')
    prompts_enhanced = gen_enhanced.generate_batch(timestamp)

    print("\nEnhanced prompts:")
    for i, prompt in enumerate(prompts_enhanced):
        print(f"  {i+1}. {prompt}")

    print("\nTest passed!")
