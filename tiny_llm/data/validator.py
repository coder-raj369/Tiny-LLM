"""
Data validation utilities for Tiny LLM.
Ensures data quality and consistency.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from tiny_llm.logging_config import DATA_LOGGER


class DataValidator:
    """Validate data quality and schema."""
    
    @staticmethod
    def validate_qa_pair(pair: Dict, require_fields: List[str] = None) -> Tuple[bool, str]:
        """
        Validate a single Q&A pair.
        
        Returns:
            (is_valid, error_message)
        """
        if require_fields is None:
            require_fields = ["question", "answer"]
        
        for field in require_fields:
            if field not in pair:
                return False, f"Missing field: {field}"
            if not isinstance(pair[field], str):
                return False, f"Field {field} is not a string"
            if len(pair[field].strip()) == 0:
                return False, f"Field {field} is empty"
        
        return True, ""
    
    @staticmethod
    def check_token_counts(
        pair: Dict,
        tokenizer,
        max_question_tokens: int = 256,
        max_answer_tokens: int = 512,
    ) -> Tuple[bool, str, Dict]:
        """
        Check token counts for a Q&A pair.
        
        Returns:
            (is_valid, error_message, token_counts)
        """
        q_tokens = tokenizer.encode(pair["question"])
        a_tokens = tokenizer.encode(pair["answer"])
        
        token_counts = {
            "question_tokens": len(q_tokens),
            "answer_tokens": len(a_tokens),
            "total_tokens": len(q_tokens) + len(a_tokens),
        }
        
        if len(q_tokens) > max_question_tokens:
            return False, f"Question too long: {len(q_tokens)} > {max_question_tokens}", token_counts
        if len(a_tokens) > max_answer_tokens:
            return False, f"Answer too long: {len(a_tokens)} > {max_answer_tokens}", token_counts
        
        return True, "", token_counts
    
    @staticmethod
    def validate_dataset(
        filepath: str,
        tokenizer,
        max_question_tokens: int = 256,
        max_answer_tokens: int = 512,
    ) -> Dict:
        """
        Validate entire dataset.
        
        Returns:
            Validation report with statistics.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        valid_count = 0
        invalid_count = 0
        issues = []
        token_stats = {
            "total_question_tokens": 0,
            "total_answer_tokens": 0,
            "max_question_tokens": 0,
            "max_answer_tokens": 0,
            "oov_tokens": 0,
        }
        
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)
                    
                    # Schema validation
                    is_valid, error = DataValidator.validate_qa_pair(pair)
                    if not is_valid:
                        invalid_count += 1
                        issues.append({"line": line_num, "error": error})
                        continue
                    
                    # Token validation
                    is_valid, error, tokens = DataValidator.check_token_counts(
                        pair, tokenizer, max_question_tokens, max_answer_tokens
                    )
                    if not is_valid:
                        invalid_count += 1
                        issues.append({"line": line_num, "error": error})
                        continue
                    
                    # Update stats
                    valid_count += 1
                    token_stats["total_question_tokens"] += tokens["question_tokens"]
                    token_stats["total_answer_tokens"] += tokens["answer_tokens"]
                    token_stats["max_question_tokens"] = max(
                        token_stats["max_question_tokens"], tokens["question_tokens"]
                    )
                    token_stats["max_answer_tokens"] = max(
                        token_stats["max_answer_tokens"], tokens["answer_tokens"]
                    )
                
                except json.JSONDecodeError as e:
                    invalid_count += 1
                    issues.append({"line": line_num, "error": f"JSON parse error: {e}"})
        
        report = {
            "filepath": str(filepath),
            "total_rows": valid_count + invalid_count,
            "valid_rows": valid_count,
            "invalid_rows": invalid_count,
            "validation_rate": valid_count / (valid_count + invalid_count) if (valid_count + invalid_count) > 0 else 0,
            "token_stats": token_stats,
            "issues": issues[:10],  # First 10 issues
            "total_issues": len(issues),
        }
        
        return report
    
    @staticmethod
    def print_validation_report(report: Dict):
        """Print validation report."""
        DATA_LOGGER.info(f"\nValidation Report: {report['filepath']}")
        DATA_LOGGER.info(f"  Total rows: {report['total_rows']}")
        DATA_LOGGER.info(f"  Valid: {report['valid_rows']}")
        DATA_LOGGER.info(f"  Invalid: {report['invalid_rows']}")
        DATA_LOGGER.info(f"  Validation rate: {report['validation_rate']:.1%}")
        DATA_LOGGER.info(f"  Max question tokens: {report['token_stats']['max_question_tokens']}")
        DATA_LOGGER.info(f"  Max answer tokens: {report['token_stats']['max_answer_tokens']}")
        
        if report["issues"]:
            DATA_LOGGER.info(f"\n  First issues:")
            for issue in report["issues"][:3]:
                DATA_LOGGER.info(f"    Line {issue['line']}: {issue['error']}")
