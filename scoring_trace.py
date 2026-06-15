class ScoringTrace:
    def __init__(self):
        pass

    @staticmethod
    def calculate_session_trace(session_confidence, session_attention, session_focus, rolling_stress, integrity_score, gaze_deviation_ratio):
        """
        Decomposes the session score into exact contributors.
        The contributors must sum up exactly to the final session score.
        """
        # Base positive components (maximum total 100)
        attention_contrib = session_attention * 40.0
        pose_contrib = session_focus * 20.0
        comm_contrib = session_confidence * 40.0
        
        # Penalties
        gaze_penalty = gaze_deviation_ratio * 10.0
        stress_penalty = rolling_stress * 20.0
        
        # Calculate base score before integrity
        base_score = attention_contrib + pose_contrib + comm_contrib - gaze_penalty - stress_penalty
        base_score = max(0.0, min(100.0, base_score))
        
        # Integrity penalty is scaling reduction
        integrity_factor = integrity_score / 100.0
        final_score = base_score * integrity_factor
        integrity_penalty = base_score - final_score
        
        # Adjust components so that their sum equals round(final_score)
        attn = round(attention_contrib)
        pose = round(pose_contrib)
        comm = round(comm_contrib)
        gaze = round(gaze_penalty)
        stress = round(stress_penalty)
        integ = round(integrity_penalty)
        
        calculated_final = attn + pose + comm - gaze - stress - integ
        target_final = int(round(final_score))
        
        # Handle rounding adjustments to ensure mathematical equality
        diff = target_final - calculated_final
        if diff != 0:
            if attn >= pose and attn >= comm:
                attn += diff
            elif comm >= pose:
                comm += diff
            else:
                pose += diff
                
        final_score_clamped = max(0, min(100, target_final))
        
        contributors = {
            "Attention Consistency": attn,
            "Stable Head Pose": pose,
            "Strong Communication": comm,
            "Frequent Gaze Deviation": -gaze,
            "Elevated Stress": -stress,
            "Integrity Penalty": -integ
        }
        
        return final_score_clamped, contributors
