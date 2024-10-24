import React, { useState } from 'react';
import PropTypes from 'prop-types';
import "./../../css/ChanakyaQuiz.css";

const allQuestions = [
  {
    question: "What does Chanakya say is the greatest wealth?",
    answers: ["A) Gold", "B) Knowledge", "C) Land", "D) Power"],
    correct: "B) Knowledge",
  },
  {
    question: "Who(king) insulted Chanakya, ordering him to be thrown out of the assembly?",
    answers: ["A) Dhanananda", "B) Bhimadeva", "C) Chandragupta", "D) Mahapadmananda"],
    correct: "A) Dhanananda",
  },
  {
    question: "According to Chanakya, what is the greatest enemy of a person?",
    answers: ["A) Pride", "B) Greed", "C) Ignorance", "D) Laziness"],
    correct: "A) Pride",
  },
  {
    question: "According to Chanakya, what should one do to overcome difficulties?",
    answers: ["A) Give up", "B) Blame others", "C) Seek wisdom", "D) Ignore them"],
    correct: "C) Seek wisdom",
  },
  {
    question: "Which of the following does Chanakya suggest as the most important aspect of leadership?",
    answers: ["A) Strictness", "B) Generosity", "C) Intelligence", "D) Both B and C"],
    correct: "D) Both B and C",
  },
  {
    question: "What does Chanakya consider the root of all evil?",
    answers: ["A) Ignorance", "B) Desire", "C) Anger", "D) Jealousy"],
    correct: "B) Desire",
  },
  {
    question: "Chanakya advises to keep a safe distance from which type of person?",
    answers: ["A) Wise", "B) Honest", "C) Foolish", "D) Rich"],
    correct: "C) Foolish",
  },
 
  {
    question: "What does Chanakya say about friendship with wicked individuals?",
    answers: ["A) It is beneficial", "B) It is dangerous", "C) It is necessary", "D) It is optional"],
    correct: "B) It is dangerous",
  },
  {
    question: "According to Chanakya, what is the importance of efficient management of resources?",
    answers: ["A) It promotes the welfare of the citizens", "B) It ensures economic stability of the state ", "C) It creates problems", "D) It removes all problems in the kingdom."],
    correct: "A) It promotes the welfare of the citizens",
  },
  {
    question: "Who played a vital role in Chandragupta's victory over Nanda family in Magadha Dynasty?",
    answers: ["A) Amarsimha", "B) Chanakya", "C) Kalidas", "D) Harisena"],
    correct: "B) Chanakya",
  },
  {
    question: "What does Chanakya recommend to gain from others?",
    answers: ["A) Wealth", "B) Power", "C) Knowledge", "D) Fear"],
    correct: "C) Knowledge",
  },
  {
    question: "Chanakya suggests what as the best way to handle enemies?",
    answers: ["A) Ignore them", "B) Defeat them", "C) Make them friends", "D) Avoid them"],
    correct: "C) Make them friends",
  },
  {
    question: "According to Chanakya, what leads to one's downfall?",
    answers: ["A) Pride", "B) Humility", "C) Generosity", "D) Kindness"],
    correct: "A) Pride",
  },
  {
    question: "What does Chanakya consider more valuable than money?",
    answers: ["A) Health", "B) Power", "C) Knowledge", "D) Fame"],
    correct: "C) Knowledge",
  },
  {
    question: "What is Chanakya’s advice regarding decision-making?",
    answers: ["A) Decide quickly", "B) Consult others", "C) Avoid decisions", "D) Think deeply"],
    correct: "D) Think deeply",
  },
  {
    question: "Chanakya advises which of the following for maintaining good relationships?",
    answers: ["A) Fear", "B) Jealousy", "C) Trust", "D) Control"],
    correct: "C) Trust",
  },
  {
    question: "According to Chanakya, what should be avoided to maintain one's honor?",
    answers: ["A) Greed", "B) Patience", "C) Wisdom", "D) Humility"],
    correct: "A) Greed",
  },
  {
    question: "What is Chanakya’s view on self-discipline?",
    answers: ["A) It is unnecessary", "B) It is essential", "C) It is difficult", "D) It is optional"],
    correct: "B) It is essential",
  },
  {
    question: "Chanakya suggests that a wise person should be",
    answers: ["A) Proud", "B) Humble", "C) Wealthy", "D) Fearful"],
    correct: "B) Humble",
  },
  {
    question: "According to Chanakya, how should one approach learning?",
    answers: ["A) Learn from everyone", "B) Ignore others", "C) Be selective", "D) Avoid learning"],
    correct: "A) Learn from everyone",
  }

];

const getRandomQuestions = (questions, num) => {
  const shuffled = questions.sort(() => 0.5 - Math.random());
  return shuffled.slice(0, num);
};

const ChanakyaQuiz = () => {
  const [started, setStarted] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [showResult, setShowResult] = useState(false);

  const startQuiz = () => {
    const numberOfQuestions = 10;
    const selectedQuestions = getRandomQuestions(allQuestions, numberOfQuestions);
    setQuestions(selectedQuestions);
    setUserAnswers(Array(numberOfQuestions).fill(null));
    setStarted(true);
  };

  const handleAnswerClick = (answer) => {
    const newAnswers = [...userAnswers];
    newAnswers[currentQuestionIndex] = answer;
    setUserAnswers(newAnswers);
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSkip = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const finishQuiz = () => {
    setShowResult(true);
  };

  const calculateScore = () => {
    return userAnswers.reduce((score, answer, index) => {
      if (answer === questions[index].correct) {
        return score + 1;
      }
      return score;
    }, 0);
  };

  return (
    <div className="quiz">
      {!started ? (
        <>
          <h1>Ready to attempt Quiz</h1>
          <button className="start-button" onClick={startQuiz}>Start Quiz</button>
        </>
      ) : showResult ? (
        <Result 
          score={calculateScore()} 
          questions={questions} 
          userAnswers={userAnswers} 
        />
      ) : (
        <div>
          <div className="question-number">
            Question {currentQuestionIndex + 1} of {questions.length}
          </div>
          <Question question={questions[currentQuestionIndex].question} />
          <ul>
            {questions[currentQuestionIndex].answers.map((answer, index) => (
              <AnswerOption
                key={index}
                answer={answer}
                onClick={() => handleAnswerClick(answer)}
                isSelected={userAnswers[currentQuestionIndex] === answer}
              />
            ))}
          </ul>
          <div>
            <button className="quizbuttons" onClick={handlePrevious} disabled={currentQuestionIndex === 0}>Previous</button>
            {currentQuestionIndex < questions.length - 1 ? (
              <button className="quizbuttons" onClick={handleNext}>Next</button>
            ) : (
              <button className="quizbuttons" onClick={finishQuiz}>Finish</button>
            )}
            <button className="quizbuttons" onClick={handleSkip}>Skip</button>
          </div>
        </div>
      )}
    </div>
  );
};

const Question = ({ question }) => {
  return (
    <div className="question">
      <h2>{question}</h2>
    </div>
  );
};

Question.propTypes = {
  question: PropTypes.string.isRequired,
};

const AnswerOption = ({ answer, onClick, isSelected }) => {
  return (
    <li 
      className={`answer-option ${isSelected ? 'selected' : ''}`} 
      onClick={onClick}
    >
      {answer}
    </li>
  );
};

AnswerOption.propTypes = {
  answer: PropTypes.string.isRequired,
  onClick: PropTypes.func.isRequired,
  isSelected: PropTypes.bool.isRequired,
};

const Result = ({ score, questions, userAnswers }) => {
  return (
    <div className="result">
      <h2>Your Score: {score} / {questions.length}</h2>
      <h3>Answers:</h3>
      <ul className="anstop">
        {questions.map((question, index) => (
          <li className={`anslist ${userAnswers[index] === question.correct ? 'correct' : 'wrong'}`} key={index}>
            <strong>Q:</strong> {question.question}
            <br />
            <strong>Your Answer:</strong> {userAnswers[index] || "Skipped"}
            <br />
            <strong>Correct Answer:</strong> {question.correct}
          </li>
        ))}
      </ul>
    </div>
  );
};

Result.propTypes = {
  score: PropTypes.number.isRequired,
  questions: PropTypes.arrayOf(
    PropTypes.shape({
      question: PropTypes.string.isRequired,
      answers: PropTypes.arrayOf(PropTypes.string).isRequired,
      correct: PropTypes.string.isRequired,
    })
  ).isRequired,
  userAnswers: PropTypes.arrayOf(PropTypes.string).isRequired,
};

export default ChanakyaQuiz;
