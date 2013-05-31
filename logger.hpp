#ifndef LOGGER_H
#define LOGGER_H

#include <pthread.h>
#include <iostream>

class Logger;

class LoggerSink {
	private:
		Logger &m_logger;
		bool m_active;
	public:
		LoggerSink (Logger& logger, bool active) : m_logger(logger), m_active(active) { }
		Logger& getLogger () { return m_logger; }
		bool active () { return m_active; }
		void deactivate () { }
		void run (std::ostream& stream, const char s) { stream << "active = " << m_active << s; }
};

class Logger {
	private:
		std::ostream* m_stream;
		std::mutex m_mutex;
		char m_separator;
		int m_level;
	public:
		Logger () : m_stream(&std::clog), m_separator(' '), m_level(0) { }
		Logger (std::ostream& o) : m_stream(&o), m_separator(' '), m_level(0) { }
		~Logger () { }
		std::ostream& stream () { return *m_stream; }
		std::ostream& stream (std::ostream& s) { m_stream = &s; return *m_stream; }
		void lock () { m_mutex.lock(); }
		void unlock () { m_mutex.unlock(); }
		void deactivate () { }
		Logger& getLogger () { return *this; }
		char separator () { return m_separator; }
		void setSeparator (const char s) { m_separator = s; }
		void setVerbosity (int l) { m_level = l; }
		LoggerSink out (int l = 0) { return LoggerSink(*this, l<=m_level); }
		void run (std::ostream& stream, const char s) { }
		bool active () { return true; }
};

template <typename Helper, typename T>
class LoggerHelperBase {
	protected:
		Helper& m_helper;
		const T& m_value;
		bool m_active;
	public:
		LoggerHelperBase (Helper& helper, const T& value)
			: m_helper(helper)
			  , m_value(value)
			  , m_active(helper.active()) { }

		void deactivate () { m_active = false; }
};

template <typename Helper, typename T>
class LoggerHelper : public LoggerHelperBase<Helper, T> {
	public:
		typedef LoggerHelperBase<Helper, T> B;

		LoggerHelper (Helper& helper, const T& value)
			: LoggerHelperBase<Helper, T>(helper, value) {
				helper.deactivate();
			}

		~LoggerHelper () {
			if (B::m_active) {
				Logger& logger = getLogger();
				logger.lock();
				run_last(logger.stream(), logger.separator());
				logger.stream() << std::endl;
				logger.unlock();
			}
		}

		bool active () { return B::m_active; }

		void run_last (std::ostream& stream, const char s) {
			B::m_helper.run(stream, s);
			stream << B::m_value;
		}

		void run (std::ostream& stream, const char s) {
			B::m_helper.run(stream, s);
			stream << B::m_value << s;
		}

		Logger& getLogger () { return B::m_helper.getLogger(); }
};

template <typename T1>
LoggerHelper<Logger, T1> operator<< (Logger& logger, const T1& value) {
	return LoggerHelper<Logger, T1>(logger, value);
}

template <typename T1>
LoggerHelper<LoggerSink, T1> operator<< (LoggerSink sink, const T1& value) {
	return LoggerHelper<LoggerSink, T1>(sink, value);
}

template <typename Helper, typename T1, typename T2>
LoggerHelper<LoggerHelper<Helper, T1>, T2> operator<< (LoggerHelper<Helper, T1> helper, const T2& value) {
	return LoggerHelper<LoggerHelper<Helper, T1>, T2>(helper, value);
}


#endif // LOGGER_H


